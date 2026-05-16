"""MCP-backed tool source over the official SDK stdio transport."""

from __future__ import annotations

import asyncio
import inspect
import json
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolResult,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._config import McpConfig, MCPServerConfig
from design_research_agents.tools._policy import ToolPolicy


class McpProtocolError(RuntimeError):
    """Raised for MCP transport/protocol errors."""


class _SdkStdioMcpClient:
    """Synchronous facade over the official MCP Python SDK stdio client."""

    def __init__(self, *, server: MCPServerConfig, policy: ToolPolicy) -> None:
        """Initialize one stdio MCP client for the configured server.

        Args:
            server: Server configuration to launch and connect to.
            policy: Tool policy used to sanitize subprocess execution.
        """
        self._server = server
        self._policy = policy
        self._stderr_lines: deque[str] = deque(maxlen=32)

    def list_tools(self) -> list[dict[str, object]]:
        """Fetch raw tool descriptors from the remote MCP server.

        Returns:
            Parsed ``tools/list`` entries returned by the server.

        Raises:
            McpProtocolError: If the server response is malformed.
        """

        async def _list_tools() -> list[dict[str, object]]:
            anyio_module, client_session_cls, stdio_module, types_module = _import_mcp_sdk_modules()
            del anyio_module
            params = self._stdio_parameters(stdio_module=stdio_module)
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errlog:
                try:
                    async with self._stdio_client(stdio_module=stdio_module, params=params, errlog=errlog) as streams:
                        read_stream, write_stream = streams
                        async with client_session_cls(
                            read_stream,
                            write_stream,
                            read_timeout_seconds=timedelta(seconds=self._server.timeout_s),
                            client_info=types_module.Implementation(
                                name="design-research-agents",
                                version="0.3.0",
                            ),
                        ) as session:
                            await session.initialize()
                            listed = await session.list_tools()
                            tools = [_dump_model(tool) for tool in listed.tools]
                            return tools
                finally:
                    self._record_errlog(errlog)

        return cast(list[dict[str, object]], self._run(_list_tools))

    def call_tool(self, *, tool_name: str, arguments: Mapping[str, object]) -> dict[str, object]:
        """Invoke one remote MCP tool and return its raw result envelope.

        Args:
            tool_name: Remote tool name to invoke.
            arguments: JSON-serializable argument mapping for the tool.

        Returns:
            Parsed ``tools/call`` result payload.

        Raises:
            McpProtocolError: If the server response is malformed or reports an error.
        """

        async def _call_tool() -> dict[str, object]:
            anyio_module, client_session_cls, stdio_module, types_module = _import_mcp_sdk_modules()
            del anyio_module
            params = self._stdio_parameters(stdio_module=stdio_module)
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errlog:
                try:
                    async with self._stdio_client(stdio_module=stdio_module, params=params, errlog=errlog) as streams:
                        read_stream, write_stream = streams
                        async with client_session_cls(
                            read_stream,
                            write_stream,
                            read_timeout_seconds=timedelta(seconds=self._server.timeout_s),
                            client_info=types_module.Implementation(
                                name="design-research-agents",
                                version="0.3.0",
                            ),
                        ) as session:
                            await session.initialize()
                            result = await session.call_tool(tool_name, dict(arguments))
                            return _dump_model(result)
                finally:
                    self._record_errlog(errlog)

        return cast(dict[str, object], self._run(_call_tool))

    def record_stderr(self, text: str) -> None:
        """Record one or more stderr lines emitted by the SDK-managed subprocess."""
        for raw_line in text.splitlines():
            normalized = raw_line.rstrip()
            if normalized:
                self._stderr_lines.append(normalized)

    def _stderr_preview(self) -> str:
        """Return a bounded stderr preview captured by the SDK transport."""
        if not self._stderr_lines:
            return ""
        preview = "\n".join(self._stderr_lines)
        if len(preview) <= 2_000:
            return preview
        return f"...{preview[-1_997:]}"

    def _record_errlog(self, errlog: Any) -> None:
        """Read captured SDK subprocess stderr from a file handle."""
        try:
            errlog.seek(0)
            text = errlog.read()
        except (OSError, AttributeError):
            return
        if isinstance(text, str):
            self.record_stderr(text)

    def _stdio_client(self, *, stdio_module: Any, params: Any, errlog: Any) -> Any:
        """Open an SDK stdio client across MCP SDK minor-version signatures."""
        client = stdio_module.stdio_client
        try:
            signature = inspect.signature(client)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "errlog" in signature.parameters:
            return client(params, errlog=errlog)
        return client(params)

    def _stdio_parameters(self, *, stdio_module: Any) -> Any:
        command = tuple(self._server.command)
        if not command:
            raise McpProtocolError(f"MCP server '{self._server.id}' has no command configured.")
        env = self._policy.sanitize_subprocess_env(
            allowlist=self._server.env_allowlist,
            extra_env=self._server.env,
        )
        return stdio_module.StdioServerParameters(
            command=command[0],
            args=list(command[1:]),
            env=env,
        )

    def _run(self, async_fn: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise McpProtocolError("MCP stdio calls must run outside an active asyncio event loop.")

        anyio_module, _, _, _ = _import_mcp_sdk_modules()
        try:
            self._stderr_lines.clear()
            return cast(Any, anyio_module).run(async_fn)
        except McpProtocolError:
            raise
        except Exception as exc:
            stderr = self._stderr_preview()
            suffix = f" stderr={stderr!r}" if stderr else ""
            raise McpProtocolError(f"MCP server '{self._server.id}' request failed: {exc}{suffix}") from exc

    def close(self) -> None:
        """Close the client facade.

        The SDK transport opens short-lived sessions for each operation, so no
        persistent subprocess handle is retained here.
        """
        self._stderr_lines.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Perform best-effort shutdown during interpreter garbage collection."""
        self.close()


def _import_mcp_sdk_modules() -> tuple[Any, Any, Any, Any]:
    """Import the optional official MCP SDK modules."""
    try:
        import anyio
        import mcp.types as types_module
        from mcp.client import stdio as stdio_module
        from mcp.client.session import ClientSession
    except ImportError as exc:
        raise McpProtocolError(
            "The official MCP Python SDK is required for MCP tool sources. "
            "Install it with: pip install design-research-agents[mcp]"
        ) from exc
    return anyio, ClientSession, stdio_module, types_module


def _dump_model(value: object) -> dict[str, object]:
    """Return a JSON-compatible dictionary for MCP SDK Pydantic models."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


@dataclass(slots=True, frozen=True, kw_only=True)
class _McpRoute:
    """Resolved route for one public MCP tool name."""

    server_id: str
    """Configured server id that owns this route."""
    remote_tool_name: str
    """Tool name exposed by the remote MCP server."""
    spec: ToolSpec
    """Normalized tool specification exposed locally."""


class McpToolSource:
    """ToolSource implementation for configured external MCP servers."""

    source_id = "mcp"

    def __init__(self, *, mcp_config: McpConfig, policy: ToolPolicy) -> None:
        """Initialize MCP clients and route table for configured servers.

        Args:
            mcp_config: MCP server configuration for this tool source.
            policy: Tool policy shared with each managed stdio client.
        """
        self._config = mcp_config
        self._policy = policy
        self._clients: dict[str, _SdkStdioMcpClient] = {
            server.id: _SdkStdioMcpClient(server=server, policy=policy) for server in mcp_config.servers
        }
        self._routes: dict[str, _McpRoute] = {}

    def list_tools(self) -> Sequence[ToolSpec]:
        """List tools discovered across configured MCP servers.

        Returns:
            Normalized tool specifications from every configured server.
        """
        self._refresh_routes()
        return tuple(route.spec for _, route in sorted(self._routes.items()))

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one routed MCP tool and normalize response payload.

        Args:
            tool_name: Public MCP tool name requested by the caller.
            input_dict: Tool input payload to forward to the remote server.
            request_id: Request identifier passed through the tool runtime.
            dependencies: Runtime dependency bag supplied by the tool runtime.

        Returns:
            Tool result normalized to the framework's ``ToolResult`` contract.
        """
        del request_id, dependencies
        self._refresh_routes()
        route = self._routes.get(tool_name)
        if route is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"Unknown MCP tool '{tool_name}'.",
            )

        client = self._clients.get(route.server_id)
        if client is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                error=f"MCP server '{route.server_id}' is not configured.",
            )

        try:
            call_response_payload = client.call_tool(
                tool_name=route.remote_tool_name,
                arguments=input_dict,
            )
        except Exception as exc:
            return ToolResult(tool_name=tool_name, ok=False, error=str(exc))

        is_error = bool(call_response_payload.get("isError", False))
        structured_content = call_response_payload.get("structuredContent")

        if isinstance(structured_content, Mapping):
            if {"ok", "result", "artifacts", "warnings"}.issubset(structured_content.keys()):
                # Fast path: remote tool already returns our canonical ToolResult envelope.
                return ToolResult(
                    tool_name=tool_name,
                    ok=bool(structured_content.get("ok")),
                    result=structured_content.get("result", {}),
                    artifacts=structured_content.get("artifacts", ()),
                    warnings=structured_content.get("warnings", ()),
                    error=structured_content.get("error"),
                    metadata={"server_id": route.server_id, "source": "mcp"},
                )
            return ToolResult(
                tool_name=tool_name,
                ok=not is_error,
                result=dict(structured_content),
                metadata={"server_id": route.server_id, "source": "mcp"},
            )

        content = call_response_payload.get("content")
        text_payload = ""
        if isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "text":
                    text_payload = str(item.get("text", ""))
                    break

        parsed_text: object = text_payload
        if text_payload:
            try:
                parsed_text = json.loads(text_payload)
            except json.JSONDecodeError:
                parsed_text = text_payload

        return ToolResult(
            tool_name=tool_name,
            ok=not is_error,
            result=parsed_text,
            metadata={"server_id": route.server_id, "source": "mcp"},
            error="MCP tool returned error." if is_error else None,
        )

    def _refresh_routes(self) -> None:
        """Refresh routes."""
        rebuilt: dict[str, _McpRoute] = {}
        for server in self._config.servers:
            client = self._clients[server.id]
            tools = client.list_tools()
            for tool in tools:
                remote_name = str(tool.get("name", "")).strip()
                if not remote_name:
                    continue
                description = str(tool.get("description", ""))
                input_schema = tool.get("inputSchema", {"type": "object"})
                if not isinstance(input_schema, Mapping):
                    input_schema = {"type": "object"}
                canonical_name = f"{server.id}::{remote_name}"
                spec = ToolSpec(
                    name=canonical_name,
                    description=description or f"MCP tool {remote_name}",
                    input_schema=dict(input_schema),
                    output_schema={"type": "object"},
                    metadata=ToolMetadata(
                        source="mcp",
                        side_effects=ToolSideEffects(),
                        timeout_s=server.timeout_s,
                        max_output_bytes=65_536,
                        risky=True,
                        server_id=server.id,
                    ),
                )
                rebuilt[canonical_name] = _McpRoute(
                    server_id=server.id,
                    remote_tool_name=remote_name,
                    spec=spec,
                )

        self._routes = rebuilt

    def close(self) -> None:
        """Close all managed MCP client connections."""
        for client in self._clients.values():
            client.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort cleanup of managed MCP client processes."""
        self.close()


__all__ = ["McpToolSource"]
