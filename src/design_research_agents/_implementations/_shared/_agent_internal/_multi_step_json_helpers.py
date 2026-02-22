"""Shared helper functions for multi-step JSON tool-calling agents."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents._contracts._agent import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult, ToolSpec
from design_research_agents._implementations._shared._agent_internal._result_builders import (
    build_failure_result,
)


def build_step_tools_text(*, tool_specs: Mapping[str, ToolSpec]) -> str:
    """Build formatted tools text for multi-step prompt injection.

    Args:
        tool_specs: Input value for this parameter.

    Returns:
        Computed return value.
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


def resolve_step_error(step_result: ExecutionResult) -> str:
    """Extract a stable step error message from one step result.

    Args:
        step_result: Input value for this parameter.

    Returns:
        Computed return value.
    """
    raw_error = step_result.output.get("error")
    if isinstance(raw_error, str) and raw_error.strip():
        return raw_error

    for tool_result in step_result.tool_results:
        if not tool_result.ok and isinstance(tool_result.error, str) and tool_result.error.strip():
            return tool_result.error

    return "Step execution failed."


def normalize_step_final_output(raw_tool_output: object) -> dict[str, object]:
    """Normalize one step output into a dictionary payload.

    Args:
        raw_tool_output: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if isinstance(raw_tool_output, Mapping):
        return dict(raw_tool_output)
    return {"tool_output": raw_tool_output}


def failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: list[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> ExecutionResult:
    """Build normalized multi-step failure result payload.

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
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=tool_results,
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )


__all__ = [
    "build_step_tools_text",
    "failure_result",
    "normalize_step_final_output",
    "resolve_step_error",
]
