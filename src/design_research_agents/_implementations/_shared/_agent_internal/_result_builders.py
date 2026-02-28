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
        error: Input value for this parameter.
        model_response: Input value for this parameter.
        tool_results: Input value for this parameter.
        request_id: Input value for this parameter.
        dependencies: Input value for this parameter.
        metadata: Input value for this parameter.
        output: Input value for this parameter.

    Returns:
        Computed return value.
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
