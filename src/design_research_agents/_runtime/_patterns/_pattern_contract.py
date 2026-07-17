"""Shared contract helpers for reusable orchestration patterns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult

OUTPUT_CONTRACT_VERSION = 1

MODE_TWO_SPEAKER_CONVERSATION = "two_speaker_conversation"
MODE_DEBATE = "debate"
MODE_PLAN_EXECUTE = "plan_execute"
MODE_PROPOSE_CRITIC = "propose_critic"
MODE_RALPH_LOOP = "ralph_loop"
MODE_NOMINAL_TEAM = "nominal_team"
MODE_ROUTER_DELEGATE = "router_delegate"
MODE_ROUND_BASED_COORDINATION = "round_based_coordination"
MODE_BLACKBOARD = "blackboard"
MODE_TREE_SEARCH = "tree_search"
MODE_RAG = "rag"
MODE_SIMULATED_ANNEALING = "simulated_annealing"
MODE_REINFORCEMENT_LEARNING = "reinforcement_learning"


def build_pattern_output(
    *,
    final_output: Mapping[str, object] | None,
    terminated_reason: str,
    details: Mapping[str, object] | None,
    workflow_payload: Mapping[str, object] | None,
    artifacts: object | None,
    error: object | None = None,
) -> dict[str, object]:
    """Build the canonical top-level output payload for pattern results."""
    output: dict[str, object] = {
        "final_output": dict(final_output or {}),
        "terminated_reason": terminated_reason,
        "details": dict(details or {}),
        "workflow": dict(workflow_payload or {}),
        "artifacts": _normalize_artifacts(artifacts),
    }
    if isinstance(error, str) and error.strip():
        output["error"] = error
    return output


def build_pattern_execution_result(
    *,
    success: bool,
    final_output: Mapping[str, object] | None,
    terminated_reason: str,
    details: Mapping[str, object] | None,
    workflow_payload: Mapping[str, object] | None,
    artifacts: object | None,
    request_id: str,
    dependencies: Mapping[str, object],
    mode: str,
    metadata: Mapping[str, object] | None = None,
    tool_results: Sequence[ToolResult] | None = None,
    model_response: LLMResponse | None = None,
    error: object | None = None,
    requested_mode: str | None = None,
    resolved_mode: str | None = None,
    runtime_metadata: Mapping[str, object] | None = None,
) -> ExecutionResult:
    """Build one canonical ``ExecutionResult`` for a pattern run."""
    result_metadata: dict[str, object] = {
        **dict(metadata or {}),
        "request_id": request_id,
        "dependency_keys": sorted(dependencies.keys()),
        "mode": mode,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
    }
    if isinstance(requested_mode, str) and isinstance(resolved_mode, str):
        result_metadata["runtime"] = {
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "observed_metrics": dict(runtime_metadata or {}),
        }
    return ExecutionResult(
        output=build_pattern_output(
            final_output=final_output,
            terminated_reason=terminated_reason,
            details=details,
            workflow_payload=workflow_payload,
            artifacts=artifacts,
            error=error,
        ),
        success=success,
        tool_results=list(tool_results or []),
        model_response=model_response,
        metadata=result_metadata,
    )


def _normalize_artifacts(artifacts: object | None) -> list[object]:
    """Return artifacts normalized to one deterministic list."""
    if isinstance(artifacts, list):
        return list(artifacts)
    if isinstance(artifacts, tuple):
        return list(artifacts)
    if isinstance(artifacts, Sequence) and not isinstance(artifacts, str | bytes | bytearray):
        return list(artifacts)
    return []


__all__ = [
    "MODE_BLACKBOARD",
    "MODE_DEBATE",
    "MODE_NOMINAL_TEAM",
    "MODE_PLAN_EXECUTE",
    "MODE_PROPOSE_CRITIC",
    "MODE_RAG",
    "MODE_RALPH_LOOP",
    "MODE_REINFORCEMENT_LEARNING",
    "MODE_ROUND_BASED_COORDINATION",
    "MODE_ROUTER_DELEGATE",
    "MODE_SIMULATED_ANNEALING",
    "MODE_TREE_SEARCH",
    "MODE_TWO_SPEAKER_CONVERSATION",
    "OUTPUT_CONTRACT_VERSION",
    "build_pattern_execution_result",
    "build_pattern_output",
]
