"""Shared result-construction helpers for agent implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult


def build_failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: Sequence[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> ExecutionResult:
    """Build a structured failure result with stable metadata fields.

    Args:
        error: Value supplied for ``error``.
        model_response: Value supplied for ``model_response``.
        tool_results: Value supplied for ``tool_results``.
        request_id: Value supplied for ``request_id``.
        dependencies: Value supplied for ``dependencies``.
        metadata: Value supplied for ``metadata``.
        output: Value supplied for ``output``.

    Returns:
        Result produced by this call.
    """
    return ExecutionResult(
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
