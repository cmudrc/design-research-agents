"""Multi-step ReAct-style agent built as a loop over ``SingleStepCodeToolCallingAgent``.

The agent alternates continuation checks with step execution, recording a
structured thought-action-observation memory trace and aggregating tool
results across steps.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent.implementations.single_step_code_tool_calling_agent import (
    SingleStepCodeToolCallingAgent,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.prompt_alternatives import (
    AlternativesPromptTarget,
    inject_alternatives_into_prompt_pair,
    resolve_alternatives_prompt_target,
)
from design_research_agents.agent.internal.response_schemas import (
    build_continuation_response_schema,
    clone_response_schema,
)
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.prompts import load_prompt, render_prompt
from design_research_agents.tracing import (
    emit_continuation_decision,
    emit_guardrail_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


class MultiStepCodeToolCallingAgent(Agent):
    """Agent that iterates action-observation steps until continuation stops.

    Each iteration asks the model whether to continue, then delegates one action
    step to ``SingleStepCodeToolCallingAgent`` with inherited runtime constraints. The
    loop keeps explicit ReAct-style thought-action-observation entries in memory.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        max_steps: int = 5,
        max_tool_calls_per_step: int = 5,
        execution_timeout_seconds_per_step: int = 5,
        validate_tool_input_schema: bool = False,
        normalize_generated_code_per_step: bool = False,
        stop_on_step_failure: bool = True,
        default_tools_per_step: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """Initialize a multi-step agent.

        Args:
            llm_client: LLM client used for continuation and action generation.
            tool_runtime: Tool runtime shared across all steps.
            max_steps: Maximum number of action-observation iterations.
            max_tool_calls_per_step: Tool-call limit applied to each action step.
            execution_timeout_seconds_per_step: Code execution timeout for each action step.
            validate_tool_input_schema: Whether to validate tool input schemas on each step.
            normalize_generated_code_per_step: Whether to apply conservative
                pre-validation code normalization in each step agent run.
            stop_on_step_failure: Whether to stop immediately when one step fails.
            default_tools_per_step: Optional allowed-tool config forwarded to each
                ``SingleStepCodeToolCallingAgent`` step. When omitted, all runtime tools are
                available per step.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if max_tool_calls_per_step < 1:
            raise ValueError("max_tool_calls_per_step must be >= 1.")
        if execution_timeout_seconds_per_step < 1:
            raise ValueError("execution_timeout_seconds_per_step must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._max_steps = max_steps
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._execution_timeout_seconds_per_step = execution_timeout_seconds_per_step
        self._validate_tool_input_schema = validate_tool_input_schema
        self._normalize_generated_code_per_step = normalize_generated_code_per_step
        self._stop_on_step_failure = stop_on_step_failure
        self._default_tools_per_step = (
            tuple(
                dict(default_tool)
                for default_tool in default_tools_per_step
                if isinstance(default_tool, Mapping)
            )
            if default_tools_per_step is not None
            else None
        )
        self._continuation_response_schema = build_continuation_response_schema()

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run the multi-step action-observation loop and return aggregated results.

        The run collects continuation decisions, per-step outputs, and all tool
        results while preserving memory entries that can be inspected by callers.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        trace_scope = start_trace_run(
            agent_name="MultiStepCodeToolCallingAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
        )
        prompt = _extract_prompt(normalized_input)
        max_steps = _extract_positive_int(
            input_payload=normalized_input,
            key="max_steps",
            default_value=self._max_steps,
        )
        max_tool_calls_per_step = _extract_positive_int(
            input_payload=normalized_input,
            key="max_tool_calls_per_step",
            default_value=self._max_tool_calls_per_step,
        )
        execution_timeout_seconds_per_step = _extract_positive_int(
            input_payload=normalized_input,
            key="execution_timeout_seconds_per_step",
            default_value=self._execution_timeout_seconds_per_step,
        )
        validate_tool_input_schema = _extract_boolean(
            input_payload=normalized_input,
            key="validate_tool_input_schema",
            default_value=self._validate_tool_input_schema,
        )
        normalize_generated_code_per_step = self._normalize_generated_code_per_step
        stop_on_step_failure = _extract_boolean(
            input_payload=normalized_input,
            key="stop_on_step_failure",
            default_value=self._stop_on_step_failure,
        )
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        alternatives_prompt_target = resolve_alternatives_prompt_target(
            input_payload=normalized_input
        )
        step_tools_text = _build_step_tools_text(
            tool_specs={spec.name: spec for spec in self._tool_runtime.list_tools()},
            default_tools_per_step=self._default_tools_per_step,
        )

        step_agent = SingleStepCodeToolCallingAgent(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            max_tool_calls=max_tool_calls_per_step,
            execution_timeout_seconds=execution_timeout_seconds_per_step,
            validate_tool_input_schema=validate_tool_input_schema,
            normalize_generated_code=normalize_generated_code_per_step,
            default_tools=self._default_tools_per_step,
        )

        memory: list[dict[str, object]] = [{"kind": "task", "prompt": prompt}]
        continuation_trace: list[dict[str, object]] = []
        step_outputs: list[dict[str, object]] = []
        tool_results: list[ToolResult] = []
        final_output: dict[str, object] = {}
        last_model_response: LLMResponse | None = None
        terminated_reason = "max_steps_reached"

        for step_index in range(max_steps):
            try:
                should_continue, continue_reason, continue_source, continue_response = (
                    self._llm_should_continue(
                        prompt=prompt,
                        memory=memory,
                        step_index=step_index,
                        max_steps=max_steps,
                        model=resolved_model,
                        alternatives_prompt_target=alternatives_prompt_target,
                        alternatives_text=step_tools_text,
                    )
                )
            except Exception as exc:
                finish_trace_run(trace_scope, error=str(exc))
                raise
            if continue_response is not None:
                last_model_response = continue_response
            continuation_trace.append(
                {
                    "step": step_index + 1,
                    "continue": should_continue,
                    "thought": continue_reason,
                    "reason": continue_reason,
                    "source": continue_source,
                }
            )
            memory.append(
                {
                    "kind": "thought",
                    "step": step_index + 1,
                    "continue": should_continue,
                    "text": continue_reason,
                    "source": continue_source,
                }
            )
            if not should_continue:
                terminated_reason = f"continuation_stopped:{continue_source}"
                break

            step_prompt = _build_step_prompt(
                prompt=prompt,
                memory=memory,
                step_number=step_index + 1,
            )
            step_input = dict(normalized_input)
            step_input["prompt"] = step_prompt
            step_input["max_tool_calls"] = max_tool_calls_per_step
            step_input["execution_timeout_seconds"] = execution_timeout_seconds_per_step
            step_input["validate_tool_input_schema"] = validate_tool_input_schema
            step_input["alternatives_prompt_target"] = alternatives_prompt_target

            step_request_id = f"{resolved_request_id}:step-{step_index + 1}"

            step_result = step_agent.run(
                json.dumps(step_input, indent=2, sort_keys=True, default=str),
                request_id=step_request_id,
                dependencies=resolved_dependencies,
            )
            if step_result.model_response is not None:
                last_model_response = step_result.model_response

            tool_results.extend(step_result.tool_results)
            step_output = {
                "step": step_index + 1,
                "success": step_result.success,
                "final_output": step_result.output.get("final_output", {}),
                "error": step_result.output.get("error"),
                "tool_results_count": len(step_result.tool_results),
            }
            step_outputs.append(step_output)
            memory.extend(
                [
                    {
                        "kind": "action",
                        "step": step_index + 1,
                        "generated_code": step_result.output.get("generated_code", ""),
                    },
                    {
                        "kind": "observation",
                        "step": step_index + 1,
                        "success": step_result.success,
                        "final_output": step_result.output.get("final_output", {}),
                        "error": step_result.output.get("error"),
                    },
                ]
            )

            if step_result.success:
                raw_final_output = step_result.output.get("final_output")
                if isinstance(raw_final_output, Mapping):
                    final_output = dict(raw_final_output)
                continue

            step_error = str(step_result.output.get("error", "Step execution failed."))
            if (
                _is_no_tool_call_step_failure(error=step_error)
                and not step_result.tool_results
                and bool(final_output)
            ):
                # No-op model step after successful prior observations: stop cleanly.
                step_outputs.pop()
                memory.pop()
                memory.pop()
                terminated_reason = "continuation_stopped:empty_step"
                break

            terminated_reason = "step_failure"
            if stop_on_step_failure:
                result = _failure_result(
                    error=step_error,
                    model_response=last_model_response,
                    tool_results=tool_results,
                    request_id=resolved_request_id,
                    dependencies=resolved_dependencies,
                    metadata={
                        "stage": "step_execution",
                        "terminated_reason": terminated_reason,
                        "continuation": continuation_trace,
                    },
                    output={
                        "final_output": final_output,
                        "steps_executed": len(step_outputs),
                        "step_outputs": step_outputs,
                        "memory": memory,
                        "terminated_reason": terminated_reason,
                    },
                )
                finish_trace_run(trace_scope, result=result)
                return result

        success = all(step_output["success"] is True for step_output in step_outputs)
        output: dict[str, object] = {
            "final_output": final_output,
            "steps_executed": len(step_outputs),
            "step_outputs": step_outputs,
            "memory": memory,
            "terminated_reason": terminated_reason,
        }
        result = AgentResult(
            output=output,
            success=success,
            tool_results=tool_results,
            model_response=last_model_response,
            metadata={
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "continuation": continuation_trace,
                "config": {
                    "max_steps": max_steps,
                    "max_tool_calls_per_step": max_tool_calls_per_step,
                    "execution_timeout_seconds_per_step": execution_timeout_seconds_per_step,
                    "validate_tool_input_schema": validate_tool_input_schema,
                    "normalize_generated_code_per_step": normalize_generated_code_per_step,
                    "stop_on_step_failure": stop_on_step_failure,
                    "default_tools_per_step": (
                        [dict(default_tool) for default_tool in self._default_tools_per_step]
                        if self._default_tools_per_step is not None
                        else None
                    ),
                },
            },
        )
        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Emit a deterministic stream wrapper around ``run``.

        The current implementation emits exactly one full-text delta followed by
        a completion event containing the full ``AgentResult`` payload.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Yields:
            Streaming events through completion.
        """
        result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)

    def _llm_should_continue(
        self,
        *,
        prompt: str,
        memory: Sequence[Mapping[str, object]],
        step_index: int,
        max_steps: int,
        model: str,
        alternatives_prompt_target: AlternativesPromptTarget,
        alternatives_text: str,
    ) -> tuple[bool, str, str, LLMResponse | None]:
        """Ask the model whether execution should continue to the next step.

        When model output is invalid JSON, the method falls back to deterministic
        continuation heuristics so loop behavior remains predictable.

        Args:
            prompt: User prompt text.
            memory: Current memory trace entries.
            step_index: Zero-based step index.
            max_steps: Maximum number of steps allowed.
            model: Model identifier for the call.
            alternatives_prompt_target: Prompt target for alternatives injection.
            alternatives_text: Alternatives block text for prompt injection.

        Returns:
            Tuple of continuation decision, reason, source, and model response.
        """
        system_prompt = load_prompt("multi_step_continue_system")
        user_prompt = _build_continue_prompt(
            prompt=prompt,
            memory=memory,
            step_number=step_index + 1,
        )
        system_prompt, user_prompt = inject_alternatives_into_prompt_pair(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            section_label="Allowed tools for action steps",
            alternatives_text=alternatives_text,
            target=alternatives_prompt_target,
        )
        messages = [
            LLMMessage(
                role="system",
                content=system_prompt,
            ),
            LLMMessage(
                role="user",
                content=user_prompt,
            ),
        ]
        llm_params = LLMChatParams(
            response_schema=clone_response_schema(self._continuation_response_schema),
            provider_options={"agent": "MultiStepCodeToolCallingAgent", "phase": "continuation"},
        )
        model_span_id = start_model_call(
            model=model,
            messages=messages,
            params=llm_params,
            metadata={"agent": "MultiStepCodeToolCallingAgent", "phase": "continuation"},
        )
        try:
            response = self._llm_client.chat(messages, model=model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=model)
            raise
        finish_model_call(model_span_id, response=response)
        parsed = _parse_json_mapping(response.text)
        if parsed is not None and isinstance(parsed.get("continue"), bool):
            # Ensure at least one action-observation cycle runs before stopping.
            if step_index == 0 and not bool(parsed["continue"]) and not _has_observation(memory):
                emit_guardrail_decision(
                    guardrail="continuation_first_step",
                    decision="override_continue",
                    reason="first-step guardrail",
                    details={"step": step_index + 1},
                )
                emit_continuation_decision(
                    step=step_index + 1,
                    should_continue=True,
                    reason="first-step guardrail",
                    source="guardrail",
                )
                return True, "first-step guardrail", "guardrail", response
            thought = _extract_continuation_thought(parsed)
            emit_continuation_decision(
                step=step_index + 1,
                should_continue=bool(parsed["continue"]),
                reason=thought,
                source="model",
            )
            return bool(parsed["continue"]), thought, "model", response

        fallback_decision = _fallback_should_continue(
            memory=memory,
            step_index=step_index,
            max_steps=max_steps,
        )
        emit_continuation_decision(
            step=step_index + 1,
            should_continue=fallback_decision,
            reason="fallback heuristic",
            source="fallback",
        )
        return fallback_decision, "fallback heuristic", "fallback", response


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from run input.

    Falls back to ``text`` and then a default string when missing.

    Args:
        input_payload: Normalized run input payload mapping.

    Returns:
        Prompt text for the run.
    """
    raw_prompt = input_payload.get(
        "prompt", input_payload.get("text", "Provide a concise response.")
    )
    return str(raw_prompt)


def _extract_positive_int(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: int,
) -> int:
    """Extract a positive integer option from run input.

    Invalid, missing, or boolean values resolve to the provided default.

    Args:
        input_payload: Normalized run input payload mapping.
        key: Input payload key to extract.
        default_value: Default value when extraction fails.

    Returns:
        Positive integer value for the option.
    """
    raw_value = input_payload.get(key)
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return default_value
    if isinstance(raw_value, int) and raw_value >= 1:
        return raw_value
    return default_value


def _extract_boolean(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: bool,
) -> bool:
    """Extract a boolean option from run input.

    Non-boolean values resolve to the provided default.

    Args:
        input_payload: Normalized run input payload mapping.
        key: Input payload key to extract.
        default_value: Default value when extraction fails.

    Returns:
        Boolean value for the option.
    """
    raw_value = input_payload.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    return default_value


def _build_continue_prompt(
    *,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_number: int,
) -> str:
    """Build continuation-decision prompt from task context and memory tail.

    The memory payload is intentionally truncated to recent entries to keep
    continuation prompts compact.

    Args:
        prompt: User prompt text.
        memory: Current memory trace entries.
        step_number: One-based step number.

    Returns:
        Rendered continuation prompt text.
    """
    memory_preview = json.dumps(list(memory)[-6:], sort_keys=True)
    return render_prompt(
        "multi_step_continue_user",
        variables={
            "step_number": step_number,
            "task_prompt": prompt,
            "memory_tail": memory_preview,
        },
    )


def _extract_continuation_thought(parsed: Mapping[str, object]) -> str:
    """Extract normalized continuation thought text from model JSON output.

    Args:
        parsed: Parsed JSON mapping from the model response.

    Returns:
        Normalized thought string.
    """
    thought = parsed.get("thought")
    if thought is not None:
        return str(thought)
    return "model decision"


def _build_step_prompt(
    *,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_number: int,
) -> str:
    """Build action prompt for the current step using recent memory context.

    The prompt includes a compact memory tail to ground step decisions while
    controlling token growth across longer runs.

    Args:
        prompt: User prompt text.
        memory: Current memory trace entries.
        step_number: One-based step number.

    Returns:
        Rendered action prompt text.
    """
    memory_preview = json.dumps(list(memory)[-8:], sort_keys=True)
    return render_prompt(
        "multi_step_step_user",
        variables={
            "task_prompt": prompt,
            "step_number": step_number,
            "memory_tail": memory_preview,
        },
    )


def _build_step_tools_text(
    *,
    tool_specs: Mapping[str, ToolSpec],
    default_tools_per_step: Sequence[Mapping[str, object]] | None,
) -> str:
    """Build formatted allowed-tools text for multi-step prompt injection.

    Args:
        tool_specs: Tool specs available in the runtime.
        default_tools_per_step: Optional default tools configuration.

    Returns:
        Rendered allowed-tools block text.
    """
    selected_specs: list[ToolSpec] = []
    if default_tools_per_step is None:
        selected_specs = list(tool_specs.values())
    else:
        for default_tool in default_tools_per_step:
            raw_name = default_tool.get("tool_name", default_tool.get("name"))
            if not isinstance(raw_name, str):
                continue
            normalized_name = raw_name.strip()
            if not normalized_name:
                continue
            runtime_spec = tool_specs.get(normalized_name)
            if runtime_spec is None:
                continue
            selected_specs.append(runtime_spec)

    tool_lines: list[str] = []
    for spec in selected_specs:
        tool_lines.append(
            "\n".join(
                [
                    f"- tool_name: {spec.name}",
                    f"  description: {spec.description or '(none)'}",
                    f"  input_schema: {json.dumps(spec.input_schema, sort_keys=True)}",
                ]
            )
        )
    return "\n".join(tool_lines)


def _parse_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Parse the first JSON object found in model text, if any.

    This allows tolerant parsing when the model adds explanatory text around the
    structured payload.

    Args:
        raw_text: Raw model response text.

    Returns:
        Parsed JSON mapping or ``None`` when parsing fails.
    """
    parsed_direct = _load_json_mapping(raw_text)
    if parsed_direct is not None:
        return parsed_direct

    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _load_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Load text as a JSON mapping.

    Returns ``None`` when the text is invalid JSON or not an object.

    Args:
        raw_text: Raw text to parse as JSON.

    Returns:
        Parsed JSON mapping or ``None`` when invalid.
    """
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    return dict(parsed)


def _fallback_should_continue(
    *,
    memory: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
) -> bool:
    """Fallback continuation policy used when model output is invalid JSON.

    The policy guarantees one initial step and then stops unless additional
    heuristics explicitly indicate continuation.

    Args:
        memory: Current memory trace entries.
        step_index: Zero-based step index.
        max_steps: Maximum number of steps allowed.

    Returns:
        ``True`` when execution should continue.
    """
    if step_index >= max_steps:
        return False

    # On parse failure, guarantee one first step, then stop by default.
    if step_index == 0:
        return True

    # If the last observation failed, stop.
    for entry in reversed(memory):
        if entry.get("kind") != "observation":
            continue
        if entry.get("success") is False:
            return False
        break

    return False


def _has_observation(memory: Sequence[Mapping[str, object]]) -> bool:
    """Return whether memory includes at least one observation entry.

    Observation entries are used by continuation guardrails and heuristics.

    Args:
        memory: Current memory trace entries.

    Returns:
        ``True`` when an observation entry exists, otherwise ``False``.
    """
    return any(entry.get("kind") == "observation" for entry in memory)


def _is_no_tool_call_step_failure(*, error: str) -> bool:
    """Return whether a step failed only because no tool call occurred.

    Args:
        error: Error message string to inspect.

    Returns:
        ``True`` when the error is a no-tool-call failure.
    """
    return "Generated code must call at least one tool." in error


def _failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: Sequence[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> AgentResult:
    """Build a structured failure ``AgentResult`` with consistent metadata.

    This helper keeps failure payload shape stable across all early-return paths.

    Args:
        error: Error message describing the failure.
        model_response: Model response payload, if available.
        tool_results: Tool results collected before failure.
        request_id: Request identifier for tracing.
        dependencies: Dependency payload mapping.
        metadata: Additional metadata to include in the result.
        output: Additional output payload fields.

    Returns:
        Agent result payload describing the failure.
    """
    return AgentResult(
        output={"error": error, **dict(output)},
        success=False,
        tool_results=list(tool_results),
        model_response=model_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            **dict(metadata),
        },
    )
