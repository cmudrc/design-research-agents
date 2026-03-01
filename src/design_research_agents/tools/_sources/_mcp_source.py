"""MCP-backed tool source over the stdio JSON-RPC transport."""

from __future__ import annotations

import json
import select
import subprocess
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock, Thread
from typing import TextIO

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


class _StdioMcpClient:
    """Tiny stdio JSON-RPC client used for MCP tools/list and tools/call."""

    def __init__(self, *, server: MCPServerConfig, policy: ToolPolicy) -> None:
        """Initialize one stdio MCP client for the configured server.

        Args:
            server: Server configuration to launch and connect to.
            policy: Tool policy used to sanitize subprocess execution.
        """
        self._server = server
        self._policy = policy
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._initialized = False
        self._stderr_lines: deque[str] = deque(maxlen=32)
        self._stderr_lock = Lock()
        self._stderr_thread: Thread | None = None

    def list_tools(self) -> list[dict[str, object]]:
        """Fetch raw tool descriptors from the remote MCP server.

        Returns:
            Parsed ``tools/list`` entries returned by the server.

        Raises:
            McpProtocolError: If the server response is malformed.
        """
        list_tools_response = self._request("tools/list", {})
        response_result = list_tools_response.get("result")
        if not isinstance(response_result, Mapping):
            raise McpProtocolError("tools/list response missing result object.")
        tools = response_result.get("tools")
        if not isinstance(tools, list):
            raise McpProtocolError("tools/list result missing tools array.")
        parsed: list[dict[str, object]] = []
        for item in tools:
            if isinstance(item, Mapping):
                parsed.append(dict(item))
        return parsed

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
        call_tool_response = self._request(
            "tools/call",
            {"name": tool_name, "arguments": dict(arguments)},
        )
        response_result = call_tool_response.get("result")
        if not isinstance(response_result, Mapping):
            raise McpProtocolError("tools/call response missing result object.")
        return dict(response_result)

    def _request(self, method: str, params: Mapping[str, object]) -> dict[str, object]:
        """Send one JSON-RPC request to the MCP server and read its response.

        Args:
            method: JSON-RPC method name to invoke.
            params: Request parameter mapping to serialize.

        Returns:
            Parsed JSON-RPC response payload for the matching request id.

        Raises:
            McpProtocolError: If the transport is unavailable or the server reports an error.
        """
        self._ensure_started()
        self._ensure_initialized()

        process = self._process
        if process is None or process.stdin is None:
            raise McpProtocolError("MCP server stdin is unavailable.")

        self._request_id += 1
        request_id = self._request_id
        request_payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        process.stdin.write(json.dumps(request_payload, ensure_ascii=True) + "\n")
        process.stdin.flush()

        # Match responses by JSON-RPC id because MCP servers may emit unrelated notifications.
        response_payload = self._read_response(expected_id=request_id)
        if "error" in response_payload:
            error = response_payload.get("error")
            if isinstance(error, Mapping):
                message = str(error.get("message", "Unknown MCP error"))
            else:
                message = "Unknown MCP error"
            raise McpProtocolError(f"MCP method '{method}' failed: {message}")
        return response_payload

    def _ensure_started(self) -> None:
        """Start the managed MCP subprocess on first use."""
        if self._process is not None:
            return

        with self._stderr_lock:
            self._stderr_lines.clear()

        # Launch subprocesses with an allowlisted environment to reduce accidental secret leakage.
        env = self._policy.sanitize_subprocess_env(
            allowlist=self._server.env_allowlist,
            extra_env=self._server.env,
        )
        self._process = subprocess.Popen(
            list(self._server.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        if self._process.stderr is not None:
            self._stderr_thread = Thread(
                target=self._drain_stderr,
                args=(self._process.stderr,),
                daemon=True,
                name=f"dra-mcp-stderr-{self._server.id}",
            )
            self._stderr_thread.start()

    def _drain_stderr(self, stderr: TextIO) -> None:
        """Continuously drain ``stderr`` to avoid child-process backpressure."""
        try:
            for line in iter(stderr.readline, ""):
                self._record_stderr_line(line)
        except Exception:
            # Best-effort background draining should not disrupt the client lifecycle.
            return

    def _record_stderr_line(self, line: str) -> None:
        """Store one bounded stderr line for later diagnostics."""
        normalized = line.rstrip()
        if not normalized:
            return
        with self._stderr_lock:
            self._stderr_lines.append(normalized)

    def _stderr_preview(self) -> str:
        """Return a bounded stderr preview captured by the background drain thread."""
        with self._stderr_lock:
            if not self._stderr_lines:
                return ""
            preview = "\n".join(self._stderr_lines)
        if len(preview) <= 2_000:
            return preview
        return f"...{preview[-1_997:]}"

    def _ensure_initialized(self) -> None:
        """Perform the MCP initialize and initialized handshake once.

        Raises:
            McpProtocolError: If the subprocess is unavailable during initialization.
        """
        if self._initialized:
            return
        process = self._process
        if process is None or process.stdin is None:
            raise McpProtocolError("MCP server process is not available.")

        self._request_id += 1
        init_id = self._request_id
        init_payload = {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "design-research-agents", "version": "0.1.0"},
            },
        }
        process.stdin.write(json.dumps(init_payload, ensure_ascii=True) + "\n")
        process.stdin.flush()
        _ = self._read_response(expected_id=init_id)

        # Follow initialize with initialized per MCP lifecycle expectations.
        self._request_id += 1
        initialized_id = self._request_id
        initialized_payload = {
            "jsonrpc": "2.0",
            "id": initialized_id,
            "method": "initialized",
            "params": {},
        }
        process.stdin.write(json.dumps(initialized_payload, ensure_ascii=True) + "\n")
        process.stdin.flush()
        _ = self._read_response(expected_id=initialized_id)

        self._initialized = True

    def _read_response(self, *, expected_id: int) -> dict[str, object]:
        """Read the next JSON-RPC response matching ``expected_id``.

        Args:
            expected_id: Request id that the response must match.

        Returns:
            Parsed response payload for the matching request id.

        Raises:
            McpProtocolError: If the process times out, exits, or emits invalid data.
        """
        process = self._process
        if process is None or process.stdout is None:
            raise McpProtocolError("MCP server stdout is unavailable.")

        deadline = time.monotonic() + self._server.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpProtocolError(f"Timed out waiting for MCP response from server '{self._server.id}'.")
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if not readable:
                continue
            line = process.stdout.readline()
            if line == "":
                stderr_text = self._stderr_preview()
                details = f" stderr={stderr_text!r}" if stderr_text else ""
                raise McpProtocolError(f"MCP server '{self._server.id}' closed unexpectedly.{details}")
            try:
                response_payload = json.loads(line)
            except json.JSONDecodeError:
                # Ignore non-JSON lines to tolerate noisy stderr-forwarding wrappers.
                continue
            if not isinstance(response_payload, Mapping):
                continue
            response_id = response_payload.get("id")
            if response_id != expected_id:
                continue
            return dict(response_payload)

    def close(self) -> None:
        """Terminate the managed MCP subprocess if it is running."""
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
        self._process = None
        self._initialized = False
        stderr_thread = self._stderr_thread
        if stderr_thread is not None:
            stderr_thread.join(timeout=0.1)
        self._stderr_thread = None
        with self._stderr_lock:
            self._stderr_lines.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Perform best-effort shutdown during interpreter garbage collection."""
        self.close()


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
        self._clients: dict[str, _StdioMcpClient] = {
            server.id: _StdioMcpClient(server=server, policy=policy) for server in mcp_config.servers
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
