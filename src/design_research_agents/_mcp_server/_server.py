"""Official MCP SDK server exposing runtime tools."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.tools import Toolbox


class McpServerDependencyError(ImportError):
    """Raised when the optional MCP SDK is not installed."""


class StdioMcpServer:
    """Thin compatibility wrapper around an official FastMCP stdio server."""

    def __init__(self, *, runtime: ToolRuntime | None = None) -> None:
        """Initialize the server with a runtime or default unified runtime."""
        self._server = create_mcp_server(runtime=runtime)

    def run(self) -> None:
        """Run the wrapped server over process stdio."""
        self._server.run(transport="stdio")

    def serve(self, *, stdin: object, stdout: object) -> None:
        """Serve over stdio.

        The official MCP SDK owns process stdio directly. The ``stdin`` and
        ``stdout`` parameters are accepted for backward-compatible call sites,
        but custom stream injection is no longer supported.
        """
        if stdin is not sys.stdin or stdout is not sys.stdout:
            raise RuntimeError("The official MCP SDK stdio server requires process stdin/stdout.")
        self.run()


def create_mcp_server(*, runtime: ToolRuntime | None = None) -> Any:
    """Build a FastMCP server from a tool runtime."""
    fastmcp_cls = _import_fastmcp()
    resolved_runtime = runtime or Toolbox()
    server = fastmcp_cls("design-research-agents")
    for spec in resolved_runtime.list_tools():
        runtime_tool = _build_runtime_tool(runtime=resolved_runtime, spec=spec)
        server.add_tool(
            runtime_tool,
            name=spec.name,
            description=spec.description,
            structured_output=True,
        )
        _patch_registered_input_schema(server=server, spec=spec)
    return server


def _build_runtime_tool(*, runtime: ToolRuntime, spec: ToolSpec) -> Any:
    signature = _signature_from_schema(tool_name=spec.name, input_schema=spec.input_schema)
    optional_none_keys = tuple(
        parameter.name for parameter in signature.parameters.values() if parameter.default is None
    )

    async def runtime_tool(**kwargs: object) -> dict[str, object]:
        forwarded = dict(kwargs)
        for key in optional_none_keys:
            if forwarded.get(key) is None:
                forwarded.pop(key, None)
        result = runtime.invoke(spec.name, forwarded, request_id="mcp", dependencies={})
        if not result.ok:
            raise ValueError(result.error_message or f"Tool {spec.name!r} failed.")
        return _tool_result_payload(result)

    runtime_tool.__name__ = f"tool_{_safe_identifier(spec.name)}"
    runtime_tool_any = cast(Any, runtime_tool)
    runtime_tool_any.__signature__ = signature
    return runtime_tool


def _patch_registered_input_schema(*, server: Any, spec: ToolSpec) -> None:
    """Preserve runtime-declared JSON schemas in FastMCP tool listings."""
    tool_manager = getattr(server, "_tool_manager", None)
    tools = getattr(tool_manager, "_tools", None)
    if isinstance(tools, dict) and spec.name in tools:
        tools[spec.name].parameters = dict(spec.input_schema)


def _signature_from_schema(*, tool_name: str, input_schema: Mapping[str, object]) -> inspect.Signature:
    schema_type = input_schema.get("type")
    if schema_type not in (None, "object"):
        return inspect.Signature(return_annotation=dict[str, object])

    properties = input_schema.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}

    required_raw = input_schema.get("required", ())
    required_names = (
        {item for item in required_raw if isinstance(item, str)}
        if isinstance(required_raw, Sequence) and not isinstance(required_raw, (str, bytes))
        else set()
    )

    parameters: list[inspect.Parameter] = []
    for raw_name, field_schema in properties.items():
        if not isinstance(raw_name, str):
            continue
        parameter_name = raw_name if raw_name.isidentifier() else _safe_identifier(raw_name)
        annotation = _annotation_from_schema(field_schema)
        default = inspect._empty
        if raw_name not in required_names:
            annotation = annotation | None
            default = None
        parameters.append(
            inspect.Parameter(
                parameter_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=annotation,
                default=default,
            )
        )
    try:
        return inspect.Signature(parameters=parameters, return_annotation=dict[str, object])
    except ValueError as exc:
        raise ValueError(f"Cannot expose tool {tool_name!r} through MCP: {exc}") from exc


def _annotation_from_schema(schema: object) -> Any:
    if not isinstance(schema, Mapping):
        return object
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [entry for entry in schema_type if entry != "null"]
        schema_type = non_null[0] if non_null else None
    if schema_type == "string":
        return str
    if schema_type == "number":
        return float
    if schema_type == "integer":
        return int
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list[object]
    if schema_type == "object":
        return dict[str, object]
    return object


def _tool_result_payload(result: ToolResult) -> dict[str, object]:
    return {
        "tool_name": result.tool_name,
        "ok": result.ok,
        "result": _to_jsonable(result.result),
        "artifacts": [_to_jsonable(asdict(artifact)) for artifact in result.artifacts],
        "warnings": list(result.warnings),
        "error": _to_jsonable(asdict(result.error)) if result.error is not None else None,
        "metadata": _to_jsonable(result.metadata),
    }


def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_identifier(name: str) -> str:
    token = "".join(char if (char.isalnum() or char == "_") else "_" for char in name)
    if not token or token[0].isdigit():
        token = f"tool_{token}"
    return token


def _import_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise McpServerDependencyError(
            "The official MCP Python SDK is required for the built-in MCP server. "
            "Install it with: pip install design-research-agents[mcp]"
        ) from exc
    return FastMCP


def _serve_stdio(runtime: ToolRuntime | None = None) -> None:
    """Start the official stdio MCP server."""
    StdioMcpServer(runtime=runtime).run()


__all__ = ["McpServerDependencyError", "StdioMcpServer", "create_mcp_server"]
