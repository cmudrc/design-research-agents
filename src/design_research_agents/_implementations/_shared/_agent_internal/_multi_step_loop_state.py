"""Shared loop-state helpers for multi-step agents."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents._contracts._termination import TERMINATED_MAX_STEPS_REACHED
from design_research_agents._contracts._tools import ToolResult


def build_loop_initial_state(
    *,
    prompt: str,
    include_continuation: bool,
) -> dict[str, object]:
    """Build the default loop state used by multi-step agents.

    Args:
        prompt: User task prompt for the loop run.
        include_continuation: Whether to include continuation/memory-trace fields.

    Returns:
        Initialized loop-state mapping.
    """
    state: dict[str, object] = {
        "memory": [{"kind": "task", "prompt": prompt}],
        "step_outputs": [],
        "tool_results": [],
        "final_output": {},
        "last_model_response": None,
        "terminated_reason": TERMINATED_MAX_STEPS_REACHED,
        "should_continue": True,
        "fatal_error": None,
        "fatal_metadata": {},
    }
    if include_continuation:
        state["decision_trace"] = []
        state["retrieval_trace"] = []
        state["memory_errors"] = []
    return state


def continue_loop(iteration: int, state: Mapping[str, object]) -> bool:
    """Return whether the next loop iteration should execute.

    Args:
        iteration: One-based loop iteration index.
        state: Current loop-state mapping.

    Returns:
        ``True`` when another iteration should execute.
    """
    del iteration
    if not bool(state.get("should_continue", True)):
        return False
    return state.get("fatal_error") is None


def coerce_state_records(raw_records: object) -> list[dict[str, object]]:
    """Coerce raw loop-state record payloads into mapping lists.

    Args:
        raw_records: Raw record payload from loop state.

    Returns:
        Normalized list of mapping records.
    """
    if not isinstance(raw_records, list):
        return []
    return [dict(record) for record in raw_records if isinstance(record, Mapping)]


def coerce_string_list(raw_values: object) -> list[str]:
    """Coerce one raw sequence into ``list[str]``.

    Args:
        raw_values: Raw value payload from loop state.

    Returns:
        Normalized list of strings.
    """
    if not isinstance(raw_values, list):
        return []
    return [str(value) for value in raw_values]


def coerce_tool_results(raw_tool_results: object) -> list[ToolResult]:
    """Coerce one raw sequence into ``list[ToolResult]``.

    Args:
        raw_tool_results: Raw tool result payload from loop state.

    Returns:
        Normalized list of ``ToolResult`` instances.
    """
    if not isinstance(raw_tool_results, list):
        return []
    return [tool_result for tool_result in raw_tool_results if isinstance(tool_result, ToolResult)]


def coerce_mapping(raw_mapping: object) -> dict[str, object]:
    """Coerce one raw object into ``dict[str, object]``.

    Args:
        raw_mapping: Raw mapping payload from loop state.

    Returns:
        Normalized dictionary payload.
    """
    if not isinstance(raw_mapping, Mapping):
        return {}
    return {str(key): value for key, value in raw_mapping.items()}


__all__ = [
    "build_loop_initial_state",
    "coerce_mapping",
    "coerce_state_records",
    "coerce_string_list",
    "coerce_tool_results",
    "continue_loop",
]
