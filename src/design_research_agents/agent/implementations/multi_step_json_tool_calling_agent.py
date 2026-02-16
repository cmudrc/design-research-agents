"""Multi-step ReAct-style agent built as a loop over ``SingleStepJsonToolCallingAgent``.

The agent alternates continuation checks with step execution, recording a
structured thought-action-observation memory trace and aggregating tool
results across steps.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent.implementations.single_step_json_tool_calling_agent import (
    SingleStepJsonToolCallingAgent,
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


class MultiStepJsonToolCallingAgent(Agent):
    """Agent that iterates action-observation steps until continuation stops.

    Each iteration asks the model whether to continue, then delegates one action
    step to ``SingleStepJsonToolCallingAgent``. The loop keeps explicit
    ReAct-style thought-action-observation entries in memory.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        max_steps: int = 5,
        stop_on_step_failure: bool = True,
        continuation_system_prompt: str | None = None,
        continuation_user_prompt_template: str | None = None,
        step_user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        continuation_memory_tail_items: int = 6,
        step_memory_tail_items: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a multi-step JSON tool-calling agent.

        Args:
            llm_client: LLM client used for continuation and step generation.
            tool_runtime: Tool runtime shared across all steps.
            max_steps: Maximum number of action-observation iterations.
            stop_on_step_failure: Whether to stop immediately when one step fails.
            continuation_system_prompt: Optional continuation system prompt override.
            continuation_user_prompt_template: Optional continuation user prompt template.
            step_user_prompt_template: Optional step user prompt template.
            alternatives_prompt_target: Prompt target for alternatives blocks.
            continuation_memory_tail_items: Memory tail size for continuation prompts.
            step_memory_tail_items: Memory tail size for step prompts.
            tracer: Optional explicit tracer dependency.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if continuation_memory_tail_items < 1:
            raise ValueError("continuation_memory_tail_items must be >= 1.")
        if step_memory_tail_items < 1:
            raise ValueError("step_memory_tail_items must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self._max_steps = max_steps
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
            default_prompt_name="multi_step_json_step_user",
            field_name="step_user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(
            alternatives_prompt_target
        )
        self._continuation_memory_tail_items = continuation_memory_tail_items
        self._step_memory_tail_items = step_memory_tail_items
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
            agent_name="MultiStepJsonToolCallingAgent",
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
        )

        step_agent = SingleStepJsonToolCallingAgent(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            alternatives_prompt_target=alternatives_prompt_target,
            tracer=self._tracer,
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

            step_prompt = build_step_prompt(
                prompt=prompt,
                memory=memory,
                step_number=step_index + 1,
                prompt_template=self._step_user_prompt_template,
                memory_tail_items=self._step_memory_tail_items,
            )
            step_request_id = f"{resolved_request_id}:step-{step_index + 1}"

            step_result = step_agent.run(
                step_prompt,
                request_id=step_request_id,
                dependencies=resolved_dependencies,
            )
            if step_result.model_response is not None:
                last_model_response = step_result.model_response

            tool_results.extend(step_result.tool_results)
            raw_tool_output = step_result.output.get("tool_output")
            step_final_output = _normalize_step_final_output(raw_tool_output)
            step_error = _resolve_step_error(step_result)
            step_output = {
                "step": step_index + 1,
                "success": step_result.success,
                "final_output": step_final_output,
                "tool_name": step_result.output.get("tool_name"),
                "tool_input": step_result.output.get("tool_input", {}),
                "error": step_error,
                "tool_results_count": len(step_result.tool_results),
            }
            step_outputs.append(step_output)
            memory.extend(
                [
                    {
                        "kind": "action",
                        "step": step_index + 1,
                        "tool_name": step_result.output.get("tool_name"),
                        "tool_input": step_result.output.get("tool_input", {}),
                    },
                    {
                        "kind": "observation",
                        "step": step_index + 1,
                        "success": step_result.success,
                        "final_output": step_final_output,
                        "error": step_error,
                    },
                ]
            )

            if step_result.success:
                final_output = step_final_output
                continue

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
                    "stop_on_step_failure": stop_on_step_failure,
                    "alternatives_prompt_target": alternatives_prompt_target,
                    "continuation_memory_tail_items": self._continuation_memory_tail_items,
                    "step_memory_tail_items": self._step_memory_tail_items,
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
        system_prompt = self._continuation_system_prompt
        user_prompt = build_continue_prompt(
            prompt=prompt,
            memory=memory,
            step_number=step_index + 1,
            prompt_template=self._continuation_user_prompt_template,
            memory_tail_items=self._continuation_memory_tail_items,
        )
        system_prompt, user_prompt = inject_alternatives_into_prompt_pair(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            section_label="Available tools for action steps",
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
            provider_options={"agent": "MultiStepJsonToolCallingAgent", "phase": "continuation"},
        )
        model_span_id = start_model_call(
            model=model,
            messages=messages,
            params=llm_params,
            metadata={"agent": "MultiStepJsonToolCallingAgent", "phase": "continuation"},
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


def _build_step_tools_text(*, tool_specs: Mapping[str, ToolSpec]) -> str:
    """Build formatted tools text for multi-step prompt injection.

    Args:
        tool_specs: Tool specs available in the runtime.

    Returns:
        Rendered tools block text.
    """
    tool_lines: list[str] = []
    for spec in tool_specs.values():
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


def _resolve_step_error(step_result: AgentResult) -> str:
    """Extract a stable step error message from one step result."""
    raw_error = step_result.output.get("error")
    if isinstance(raw_error, str) and raw_error.strip():
        return raw_error

    for tool_result in step_result.tool_results:
        if not tool_result.ok and isinstance(tool_result.error, str) and tool_result.error.strip():
            return tool_result.error

    return "Step execution failed."


def _normalize_step_final_output(raw_tool_output: object) -> dict[str, object]:
    """Normalize one step output into a dictionary payload."""
    if isinstance(raw_tool_output, Mapping):
        return dict(raw_tool_output)
    return {"tool_output": raw_tool_output}


def _failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: list[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> AgentResult:
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=tool_results,
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )
