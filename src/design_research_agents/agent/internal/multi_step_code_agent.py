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
from design_research_agents.agent.internal.input_parsing import (
    extract_boolean as _extract_boolean,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_positive_int as _extract_positive_int,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents.agent.internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.multi_step_common import (
    build_continue_prompt,
    build_step_prompt,
    extract_continuation_thought,
    fallback_should_continue,
    has_observation,
)
from design_research_agents.agent.internal.multi_step_memory import (
    retrieve_memory_context,
    write_memory_observation,
)
from design_research_agents.agent.internal.prompt_alternatives import (
    AlternativesPromptTarget,
    inject_alternatives_into_prompt_pair,
    normalize_alternatives_prompt_target,
)
from design_research_agents.agent.internal.prompt_overrides import resolve_prompt_text
from design_research_agents.agent.internal.response_schemas import (
    build_continuation_response_schema,
    clone_response_schema,
)
from design_research_agents.agent.internal.result_builders import build_failure_result
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
from design_research_agents.contracts.memory import MemoryStore
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.tracing import (
    Tracer,
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
        continuation_system_prompt: str | None = None,
        continuation_user_prompt_template: str | None = None,
        step_user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        continuation_memory_tail_items: int = 6,
        step_memory_tail_items: int = 8,
        memory_store: MemoryStore | None = None,
        memory_namespace: str = "default",
        memory_read_top_k: int = 4,
        memory_write_observations: bool = True,
        tracer: Tracer | None = None,
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
            continuation_system_prompt: Optional continuation system prompt override.
            continuation_user_prompt_template: Optional continuation user prompt template.
            step_user_prompt_template: Optional step user prompt template.
            alternatives_prompt_target: Prompt target for alternatives blocks.
            continuation_memory_tail_items: Memory tail size for continuation prompts.
            step_memory_tail_items: Memory tail size for step prompts.
            memory_store: Optional persistent memory store for retrieval/write-back.
            memory_namespace: Namespace partition used for memory reads/writes.
            memory_read_top_k: Number of memory matches retrieved per step.
            memory_write_observations: Whether to persist per-step observations.
            tracer: Optional explicit tracer dependency.

        Raises:
            Exception: Raised when execution fails.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if max_tool_calls_per_step < 1:
            raise ValueError("max_tool_calls_per_step must be >= 1.")
        if execution_timeout_seconds_per_step < 1:
            raise ValueError("execution_timeout_seconds_per_step must be >= 1.")
        if continuation_memory_tail_items < 1:
            raise ValueError("continuation_memory_tail_items must be >= 1.")
        if step_memory_tail_items < 1:
            raise ValueError("step_memory_tail_items must be >= 1.")
        if memory_read_top_k < 1:
            raise ValueError("memory_read_top_k must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self._max_steps = max_steps
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._execution_timeout_seconds_per_step = execution_timeout_seconds_per_step
        self._validate_tool_input_schema = validate_tool_input_schema
        self._normalize_generated_code_per_step = normalize_generated_code_per_step
        self._stop_on_step_failure = stop_on_step_failure
        self._continuation_system_prompt = resolve_prompt_text(
            override=continuation_system_prompt,
            default_prompt_name="multi_step_continue_system",
            field_name="continuation_system_prompt",
        )
        self._continuation_user_prompt_template = resolve_prompt_text(
            override=continuation_user_prompt_template,
            default_prompt_name="multi_step_continue_user",
            field_name="continuation_user_prompt_template",
        )
        self._step_user_prompt_template = resolve_prompt_text(
            override=step_user_prompt_template,
            default_prompt_name="multi_step_step_user",
            field_name="step_user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(
            alternatives_prompt_target
        )
        self._continuation_memory_tail_items = continuation_memory_tail_items
        self._step_memory_tail_items = step_memory_tail_items
        self._memory_store = memory_store
        self._memory_namespace = memory_namespace.strip() or "default"
        self._memory_read_top_k = memory_read_top_k
        self._memory_write_observations = memory_write_observations
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

    def run(  # noqa: C901
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

        Raises:
            Exception: Raised when execution fails.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        trace_scope = start_trace_run(
            agent_name="MultiStepCodeToolCallingAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
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
        alternatives_prompt_target = self._alternatives_prompt_target
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
            alternatives_prompt_target=alternatives_prompt_target,
            tracer=self._tracer,
        )

        memory: list[dict[str, object]] = [{"kind": "task", "prompt": prompt}]
        continuation_trace: list[dict[str, object]] = []
        retrieval_trace: list[dict[str, object]] = []
        memory_errors: list[str] = []
        step_outputs: list[dict[str, object]] = []
        tool_results: list[ToolResult] = []
        final_output: dict[str, object] = {}
        last_model_response: LLMResponse | None = None
        terminated_reason = "max_steps_reached"

        for step_index in range(max_steps):
            retrieved_context, retrieved_matches, retrieval_error = retrieve_memory_context(
                memory_store=self._memory_store,
                namespace=self._memory_namespace,
                top_k=self._memory_read_top_k,
                task_prompt=prompt,
                memory=memory,
                memory_tail_items=self._continuation_memory_tail_items,
            )
            if retrieval_error is not None:
                memory_errors.append(f"read(step {step_index + 1}): {retrieval_error}")
            retrieval_trace.append(
                {
                    "step": step_index + 1,
                    "count": len(retrieved_matches),
                    "namespace": self._memory_namespace,
                }
            )

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
                        retrieved_context=retrieved_context,
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

            step_prompt = build_step_prompt(
                prompt=prompt,
                memory=memory,
                step_number=step_index + 1,
                prompt_template=self._step_user_prompt_template,
                memory_tail_items=self._step_memory_tail_items,
                retrieved_context=retrieved_context,
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

            if self._memory_write_observations:
                memory_write_error = write_memory_observation(
                    memory_store=self._memory_store,
                    namespace=self._memory_namespace,
                    payload={
                        "task": prompt,
                        "step": step_index + 1,
                        "thought": continue_reason,
                        "selected_action": _summarize_code_action(
                            generated_code=step_result.output.get("generated_code", "")
                        ),
                        "observation_summary": _summarize_observation(
                            final_output=step_result.output.get("final_output"),
                            error=step_result.output.get("error"),
                        ),
                        "success": step_result.success,
                    },
                    metadata={
                        "kind": "multi_step_observation",
                        "agent": "MultiStepCodeToolCallingAgent",
                        "step": step_index + 1,
                        "success": step_result.success,
                    },
                )
                if memory_write_error is not None:
                    memory_errors.append(f"write(step {step_index + 1}): {memory_write_error}")

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
                    "alternatives_prompt_target": alternatives_prompt_target,
                    "continuation_memory_tail_items": self._continuation_memory_tail_items,
                    "step_memory_tail_items": self._step_memory_tail_items,
                    "memory_namespace": self._memory_namespace,
                    "memory_read_top_k": self._memory_read_top_k,
                    "memory_write_observations": self._memory_write_observations,
                    "default_tools_per_step": (
                        [dict(default_tool) for default_tool in self._default_tools_per_step]
                        if self._default_tools_per_step is not None
                        else None
                    ),
                },
                "memory": {
                    "enabled": self._memory_store is not None,
                    "retrieval_trace": retrieval_trace,
                    "errors": memory_errors,
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
        retrieved_context: str,
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
            retrieved_context: Retrieved memory context block for this step.

        Returns:
            Tuple of continuation decision, reason, source, and model response.

        Raises:
            Exception: Raised when execution fails.
        """
        system_prompt = self._continuation_system_prompt
        user_prompt = build_continue_prompt(
            prompt=prompt,
            memory=memory,
            step_number=step_index + 1,
            prompt_template=self._continuation_user_prompt_template,
            memory_tail_items=self._continuation_memory_tail_items,
            retrieved_context=retrieved_context,
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
            if step_index == 0 and not bool(parsed["continue"]) and not has_observation(memory):
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
            thought = extract_continuation_thought(parsed)
            emit_continuation_decision(
                step=step_index + 1,
                should_continue=bool(parsed["continue"]),
                reason=thought,
                source="model",
            )
            return bool(parsed["continue"]), thought, "model", response

        fallback_decision = fallback_should_continue(
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


def _is_no_tool_call_step_failure(*, error: str) -> bool:
    """Return whether a step failed only because no tool call occurred.

    Args:
        error: Error message string to inspect.

    Returns:
        ``True`` when the error is a no-tool-call failure.
    """
    return "Generated code must call at least one tool." in error


def _summarize_code_action(*, generated_code: object) -> str:
    """Return compact generated-code summary for memory write-back.

    Args:
        generated_code: Generated code payload.

    Returns:
        Compact summary string.
    """
    if not isinstance(generated_code, str):
        return ""
    normalized = generated_code.strip()
    if not normalized:
        return ""
    lines = normalized.splitlines()
    preview = " ".join(line.strip() for line in lines[:2] if line.strip())
    if len(preview) > 320:
        return preview[:317] + "..."
    return preview


def _summarize_observation(*, final_output: object, error: object) -> str:
    """Return compact observation summary for memory write-back.

    Args:
        final_output: Final output payload.
        error: Optional error payload.

    Returns:
        Compact summary string.
    """
    if isinstance(error, str) and error.strip():
        return f"error: {error.strip()}"
    if isinstance(final_output, Mapping):
        serialized = json.dumps(dict(final_output), ensure_ascii=True, sort_keys=True)
    else:
        serialized = str(final_output)
    normalized = serialized.strip()
    if len(normalized) > 320:
        return normalized[:317] + "..."
    return normalized


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
    """Run failure result.

    Args:
        error: Parameter value.
        model_response: Parameter value.
        tool_results: Parameter value.
        request_id: Parameter value.
        dependencies: Parameter value.
        metadata: Parameter value.
        output: Parameter value.

    Returns:
        The resulting value.
    """
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=tool_results,
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )
