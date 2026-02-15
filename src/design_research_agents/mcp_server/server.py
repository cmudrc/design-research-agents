"""Stdio MCP server exposing runtime tools."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import TextIO

from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tools import ToolRuntimeConfig, UnifiedToolRuntime

from .adapters import tool_result_to_mcp_payload, tool_spec_to_mcp_payload


class StdioMcpServer:
    """Minimal JSON-RPC MCP server over stdio."""

    def __init__(self, *, runtime: ToolRuntime | None = None) -> None:
        self._runtime = runtime or UnifiedToolRuntime(config=ToolRuntimeConfig())

    def serve(self, *, stdin: TextIO, stdout: TextIO) -> None:
        """Serve until stdin closes."""
        for line in stdin:
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                request = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                self._write_error(
                    stdout,
                    request_id=None,
                    code=-32700,
                    message=f"Parse error: {exc}",
                )
                continue

            response = self._handle_request(request)
            stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
            stdout.flush()

    def _handle_request(self, request: Mapping[str, object]) -> dict[str, object]:
        request_id = request.get("id")
        method = str(request.get("method", ""))
        params = request.get("params")
        if not isinstance(params, Mapping):
            params = {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "design-research-agents",
                        "version": "0.1.0",
                    },
                },
            }

        if method == "initialized":
            return {"jsonrpc": "2.0", "id": request_id, "result": None}

        if method == "tools/list":
            tools = [tool_spec_to_mcp_payload(spec) for spec in self._runtime.list_tools()]
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools},
            }

        if method == "tools/call":
            name = str(params.get("name", "")).strip()
            arguments = params.get("arguments", {})
            if not isinstance(arguments, Mapping):
                return self._error_response(
                    request_id=request_id,
                    code=-32602,
                    message="tools/call arguments must be an object.",
                )
            result = self._runtime.invoke(
                name,
                dict(arguments),
                request_id="mcp",
                dependencies={},
            )
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": tool_result_to_mcp_payload(result),
            }

        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": request_id, "result": None}

        return self._error_response(
            request_id=request_id,
            code=-32601,
            message=f"Method not found: {method}",
        )

    def _error_response(
        self,
        *,
        request_id: object,
        code: int,
        message: str,
    ) -> dict[str, object]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    def _write_error(
        self,
        stdout: TextIO,
        *,
        request_id: object,
        code: int,
        message: str,
    ) -> None:
        stdout.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": code, "message": message},
                },
                ensure_ascii=True,
            )
            + "\n"
        )
        stdout.flush()


def serve_stdio(runtime: ToolRuntime | None = None) -> None:
    """Start stdio MCP server."""
    server = StdioMcpServer(runtime=runtime)
    server.serve(stdin=sys.stdin, stdout=sys.stdout)


__all__ = ["StdioMcpServer", "serve_stdio"]
