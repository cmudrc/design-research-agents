"""Shared result-construction helpers for agent implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.agent import AgentResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolResult


def build_failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: Sequence[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> AgentResult:
    """Build a structured failure result with stable metadata fields."""
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
