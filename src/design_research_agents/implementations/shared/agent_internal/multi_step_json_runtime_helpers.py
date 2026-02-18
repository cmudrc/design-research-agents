"""Runtime helpers for ``MultiStepJsonToolCallingAgent``."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.termination import TERMINATED_MAX_STEPS_REACHED
from design_research_agents.implementations.shared.agent_internal.multi_step_json_helpers import (
    failure_result,
)
from design_research_agents.implementations.shared.agent_internal.multi_step_loop_state import (
    coerce_mapping,
    coerce_state_records,
    coerce_string_list,
    coerce_tool_results,
)


def summarize_tool_action(*, tool_name: object, tool_input: object) -> str:
    """Return a compact tool action summary for memory write-back.

    Args:
        tool_name: Tool name payload.
        tool_input: Tool input payload.

    Returns:
        Compact action summary string.
    """
    normalized_name = str(tool_name or "").strip()
    if not normalized_name:
        return ""
    if isinstance(tool_input, Mapping):
        serialized_input = json.dumps(dict(tool_input), ensure_ascii=True, sort_keys=True)
    else:
        serialized_input = str(tool_input)
    summary = f"{normalized_name} {serialized_input}".strip()
    if len(summary) > 320:
        return summary[:317] + "..."
    return summary


def summarize_observation(*, final_output: object, error: object) -> str:
    """Return a compact observation summary for memory write-back.

    Args:
        final_output: Per-step final output payload.
        error: Per-step error payload.

    Returns:
        Compact observation summary string.
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


def build_json_final_result(
    *,
    final_state: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    max_steps: int,
    stop_on_step_failure: bool,
    alternatives_prompt_target: str,
    continuation_memory_tail_items: int,
    step_memory_tail_items: int,
    memory_namespace: str,
    memory_read_top_k: int,
    memory_write_observations: bool,
    memory_store_enabled: bool,
) -> ExecutionResult:
    """Build the final normalized result for the JSON multi-step agent.

    Args:
        final_state: Final loop-state mapping.
        request_id: Request identifier for the run.
        dependencies: Dependency payload used during the run.
        max_steps: Effective max-step setting.
        stop_on_step_failure: Effective stop-on-failure setting.
        alternatives_prompt_target: Effective alternatives prompt target.
        continuation_memory_tail_items: Continuation memory tail item count.
        step_memory_tail_items: Step memory tail item count.
        memory_namespace: Memory namespace used for read/write.
        memory_read_top_k: Memory retrieval top-k setting.
        memory_write_observations: Whether observation writes were enabled.
        memory_store_enabled: Whether memory store dependency was configured.

    Returns:
        Final normalized execution result.
    """
    memory = coerce_state_records(final_state.get("memory"))
    continuation_trace = coerce_state_records(final_state.get("continuation_trace"))
    retrieval_trace = coerce_state_records(final_state.get("retrieval_trace"))
    memory_errors = coerce_string_list(final_state.get("memory_errors"))
    step_outputs = coerce_state_records(final_state.get("step_outputs"))
    tool_results = coerce_tool_results(final_state.get("tool_results"))
    final_output = coerce_mapping(final_state.get("final_output"))
    terminated_reason = str(final_state.get("terminated_reason", TERMINATED_MAX_STEPS_REACHED))
    maybe_model_response = final_state.get("last_model_response")
    last_model_response = (
        maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None
    )

    fatal_error = final_state.get("fatal_error")
    fatal_metadata = coerce_mapping(final_state.get("fatal_metadata"))
    if isinstance(fatal_error, str) and fatal_error:
        return failure_result(
            error=fatal_error,
            model_response=last_model_response,
            tool_results=tool_results,
            request_id=request_id,
            dependencies=dependencies,
            metadata={
                **fatal_metadata,
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

    success = all(step_output.get("success") is True for step_output in step_outputs)
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
            "continuation": continuation_trace,
            "config": {
                "max_steps": max_steps,
                "stop_on_step_failure": stop_on_step_failure,
                "alternatives_prompt_target": alternatives_prompt_target,
                "continuation_memory_tail_items": continuation_memory_tail_items,
                "step_memory_tail_items": step_memory_tail_items,
                "memory_namespace": memory_namespace,
                "memory_read_top_k": memory_read_top_k,
                "memory_write_observations": memory_write_observations,
            },
            "memory": {
                "enabled": memory_store_enabled,
                "retrieval_trace": retrieval_trace,
                "errors": memory_errors,
            },
        },
    )


__all__ = [
    "build_json_final_result",
    "summarize_observation",
    "summarize_tool_action",
]
