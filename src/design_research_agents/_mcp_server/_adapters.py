"""Adapters between internal tool contracts and MCP JSON-RPC payloads."""

from __future__ import annotations

import json
from dataclasses import asdict

from design_research_agents._contracts._tools import ToolResult, ToolSpec


def tool_spec_to_mcp_payload(spec: ToolSpec) -> dict[str, object]:
    """Convert one ToolSpec into MCP ``tools/list`` payload shape.

    Args:
        spec: Internal tool specification to expose over MCP.

    Returns:
        MCP-compatible ``tools/list`` entry.
    """
    return {
        "name": spec.name,
        "description": spec.description,
        "inputSchema": spec.input_schema,
    }


def tool_result_to_mcp_payload(result: ToolResult) -> dict[str, object]:
    """Convert one ToolResult into MCP ``tools/call`` response payload.

    Args:
        result: Internal tool invocation result.

    Returns:
        MCP-compatible ``tools/call`` response payload.
    """
    structured_content = {
        "tool_name": result.tool_name,
        "ok": result.ok,
        "result": result.result,
        "artifacts": [asdict(artifact) for artifact in result.artifacts],
        "warnings": list(result.warnings),
        "error": asdict(result.error) if result.error is not None else None,
        "metadata": result.metadata,
    }
    return {
        "isError": not result.ok,
        "structuredContent": structured_content,
        "content": [
            {
                "type": "text",
                "text": json.dumps(structured_content, ensure_ascii=True),
            }
        ],
    }


__all__ = ["tool_result_to_mcp_payload", "tool_spec_to_mcp_payload"]
