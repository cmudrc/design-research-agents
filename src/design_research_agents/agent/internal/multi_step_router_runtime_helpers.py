"""Runtime helpers for ``MultiStepToolRouterAgent``."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.agent.internal.multi_step_loop_state import (
    coerce_state_records,
    coerce_tool_results,
)
from design_research_agents.agent.internal.multi_step_tool_router_helpers import (
    ToolRouterStepDecision,
    failure_result,
    normalize_output_dict,
    resolve_selected_tool,
)
from design_research_agents.agent.internal.router_agent_helpers import (
    ToolAlternative,
    resolve_tool_input,
)
from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.termination import (
    TERMINATED_INVALID_ROUTE_SELECTION,
    TERMINATED_MAX_STEPS_REACHED,
    TERMINATED_STEP_FAILURE,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime
from design_research_agents.tracing import (
    emit_continuation_decision,
    emit_guardrail_decision,
    emit_router_decision,
)


def build_router_final_result(
    *,
    final_state: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    max_steps: int,
    stop_on_step_failure: bool,
    alternatives_prompt_target: str,
    step_memory_tail_items: int,
) -> ExecutionResult:
    """Build the final normalized result for the multi-step tool router.

    Args:
        final_state: Final loop-state mapping.
        request_id: Request identifier for the run.
        dependencies: Dependency payload used during the run.
        max_steps: Effective max-step setting.
        stop_on_step_failure: Effective stop-on-failure setting.
        alternatives_prompt_target: Effective alternatives prompt target.
        step_memory_tail_items: Effective step memory tail item count.

    Returns:
        Final normalized execution result.
    """
    memory = coerce_state_records(final_state.get("memory"))
    step_outputs = coerce_state_records(final_state.get("step_outputs"))
    tool_results = coerce_tool_results(final_state.get("tool_results"))
    final_output = normalize_output_dict(final_state.get("final_output"))
    terminated_reason = str(final_state.get("terminated_reason", TERMINATED_MAX_STEPS_REACHED))
    maybe_model_response = final_state.get("last_model_response")
    last_model_response = (
        maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None
    )
    fatal_error = final_state.get("fatal_error")
    fatal_metadata = normalize_output_dict(final_state.get("fatal_metadata"))
    if isinstance(fatal_error, str) and fatal_error:
        return failure_result(
            error=fatal_error,
            model_response=last_model_response,
            tool_results=tool_results,
            request_id=request_id,
            dependencies=dependencies,
            metadata=fatal_metadata,
            output={
                "final_output": final_output,
                "steps_executed": len(step_outputs),
                "step_outputs": step_outputs,
                "memory": memory,
                "terminated_reason": terminated_reason,
            },
        )

    success = all(
        step_output.get("success") is True
        for step_output in step_outputs
        if step_output.get("action") == "TOOL_CALL"
    )
    return ExecutionResult(
        output={
            "final_output": final_output,
            "steps_executed": len(step_outputs),
            "step_outputs": step_outputs,
            "memory": memory,
            "terminated_reason": terminated_reason,
        },
        success=success,
        tool_results=tool_results,
        model_response=last_model_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            "config": {
                "max_steps": max_steps,
                "stop_on_step_failure": stop_on_step_failure,
                "alternatives_prompt_target": alternatives_prompt_target,
                "step_memory_tail_items": step_memory_tail_items,
            },
        },
    )


def run_tool_call_step(
    *,
    tool_runtime: ToolRuntime,
    step_number: int,
    parsed_step: ToolRouterStepDecision,
    alternatives: list[ToolAlternative],
    normalized_input: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    memory: list[dict[str, object]],
    step_outputs: list[dict[str, object]],
    tool_results: list[ToolResult],
    final_output: dict[str, object],
    last_model_response: LLMResponse | None,
    stop_on_step_failure: bool,
) -> Mapping[str, object]:
    """Run one TOOL_CALL step branch and produce next loop state.

    Args:
        tool_runtime: Tool runtime used to invoke selected tools.
        step_number: One-based step number.
        parsed_step: Parsed router controller decision.
        alternatives: Normalized routing alternatives.
        normalized_input: Normalized run input payload.
        request_id: Resolved request identifier.
        dependencies: Normalized dependency payload mapping.
        memory: Mutable memory record list.
        step_outputs: Mutable step output list.
        tool_results: Mutable tool result list.
        final_output: Current run-level final output mapping.
        last_model_response: Most recent model response.
        stop_on_step_failure: Effective stop-on-failure setting.

    Returns:
        Next loop-state mapping.
    """
    tool_resolution = resolve_selected_tool(
        alternatives=alternatives,
        tool_names=parsed_step.tool_names,
    )
    if tool_resolution is None:
        emit_guardrail_decision(
            guardrail="route_validation",
            decision="reject",
            reason="invalid selected tool_names",
            details={"step": step_number},
        )
        return {
            "memory": memory,
            "step_outputs": step_outputs,
            "tool_results": tool_results,
            "final_output": final_output,
            "last_model_response": last_model_response,
            "terminated_reason": TERMINATED_INVALID_ROUTE_SELECTION,
            "should_continue": False,
            "fatal_error": "Step selected no valid tool route.",
            "fatal_metadata": {
                "stage": "step_decision",
                "terminated_reason": TERMINATED_INVALID_ROUTE_SELECTION,
            },
        }

    selected_tool_name, selected_tool_index = tool_resolution
    emit_router_decision(
        source=parsed_step.source,
        alternatives=[alternative.tool_name for alternative in alternatives],
        selected_tool_name=selected_tool_name,
        selected_index=selected_tool_index,
        reason=parsed_step.reason,
        parsed_route={
            "action": "TOOL_CALL",
            "tool_names": list(parsed_step.tool_names),
            "reason": parsed_step.reason,
        },
    )
    tool_input = (
        parsed_step.tool_input
        if parsed_step.tool_input is not None
        else resolve_tool_input(
            tool_name=selected_tool_name,
            input_payload=normalized_input,
        )
    )
    tool_result = tool_runtime.invoke(
        selected_tool_name,
        tool_input,
        request_id=f"{request_id}:step-{step_number}",
        dependencies=dependencies,
    )
    tool_results.append(tool_result)
    emit_continuation_decision(
        step=step_number,
        should_continue=True,
        reason=parsed_step.reason,
        source=parsed_step.source,
    )

    step_final_output = normalize_output_dict(tool_result.result)
    if tool_result.ok:
        final_output = step_final_output
    step_outputs.append(
        {
            "step": step_number,
            "action": "TOOL_CALL",
            "tool_name": selected_tool_name,
            "tool_names": list(parsed_step.tool_names),
            "tool_input": tool_input,
            "tool_output": tool_result.result,
            "reason": parsed_step.reason,
            "source": parsed_step.source,
            "success": tool_result.ok,
            "error": tool_result.error,
        }
    )
    memory.extend(
        [
            {
                "kind": "action",
                "step": step_number,
                "tool_name": selected_tool_name,
                "tool_names": list(parsed_step.tool_names),
                "tool_input": tool_input,
            },
            {
                "kind": "observation",
                "step": step_number,
                "success": tool_result.ok,
                "final_output": step_final_output,
                "error": tool_result.error,
            },
        ]
    )

    terminated_reason = TERMINATED_MAX_STEPS_REACHED
    should_continue_next = True
    fatal_error: str | None = None
    fatal_metadata: dict[str, object] = {}
    if not tool_result.ok:
        terminated_reason = TERMINATED_STEP_FAILURE
        if stop_on_step_failure:
            should_continue_next = False
            fatal_error = (
                tool_result.error.message
                if tool_result.error is not None
                else "Step tool execution failed."
            )
            fatal_metadata = {
                "stage": "step_execution",
                "terminated_reason": terminated_reason,
            }

    return {
        "memory": memory,
        "step_outputs": step_outputs,
        "tool_results": tool_results,
        "final_output": final_output,
        "last_model_response": last_model_response,
        "terminated_reason": terminated_reason,
        "should_continue": should_continue_next,
        "fatal_error": fatal_error,
        "fatal_metadata": fatal_metadata,
    }


__all__ = [
    "build_router_final_result",
    "run_tool_call_step",
]
