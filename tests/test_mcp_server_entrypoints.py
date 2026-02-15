from __future__ import annotations

import io
import json
import runpy
from collections.abc import Mapping

import pytest

from design_research_agents.contracts.tools import ToolResult, ToolSpec
from design_research_agents.mcp_server import adapters
from design_research_agents.mcp_server import cli as mcp_cli
from design_research_agents.mcp_server import server as mcp_server


class _RuntimeStub:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict[str, object], str]] = []

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec(
                name="calculator",
                description="Evaluate arithmetic expressions.",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
            ),
        )

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del dependencies
        payload = dict(input_dict)
        self.invocations.append((tool_name, payload, request_id))
        return ToolResult(tool_name=tool_name, ok=True, result={"echo": payload})


def test_stdio_server_serve_round_trip_and_errors() -> None:
    runtime = _RuntimeStub()
    server = mcp_server.StdioMcpServer(runtime=runtime)
    requests = [
        "not-json",
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "calculator", "arguments": {"value": 3}},
            }
        ),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "calculator", "arguments": []},
            }
        ),
        json.dumps({"jsonrpc": "2.0", "id": 6, "method": "unknown/method"}),
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "shutdown"}),
    ]
    stdin = io.StringIO("\n".join(requests) + "\n")
    stdout = io.StringIO()

    server.serve(stdin=stdin, stdout=stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    by_id = {response["id"]: response for response in responses if response.get("id") is not None}
    parse_error = next(response for response in responses if response.get("id") is None)

    assert parse_error["error"]["code"] == -32700
    assert "Parse error:" in parse_error["error"]["message"]

    assert by_id[1]["result"]["protocolVersion"] == "2024-11-05"
    assert by_id[2]["result"] is None
    assert by_id[3]["result"]["tools"][0]["name"] == "calculator"
    assert by_id[4]["result"]["isError"] is False
    assert by_id[4]["result"]["structuredContent"]["result"] == {"echo": {"value": 3}}
    assert by_id[5]["error"]["code"] == -32602
    assert by_id[6]["error"]["code"] == -32601
    assert by_id[7]["result"] is None
    assert runtime.invocations == [("calculator", {"value": 3}, "mcp")]


def test_mcp_adapters_preserve_structured_result() -> None:
    spec = ToolSpec(
        name="lookup",
        description="Lookup value.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    spec_payload = adapters.tool_spec_to_mcp_payload(spec)
    assert spec_payload == {
        "name": "lookup",
        "description": "Lookup value.",
        "inputSchema": {"type": "object"},
    }

    result = ToolResult(
        tool_name="lookup",
        ok=False,
        result={"value": None},
        artifacts=({"path": "artifact.txt", "mime": "text/plain"},),
        warnings=("soft warning",),
        error={"type": "ToolError", "message": "failed"},
        metadata={"source": "test"},
    )
    result_payload = adapters.tool_result_to_mcp_payload(result)

    assert result_payload["isError"] is True
    structured = result_payload["structuredContent"]
    assert structured["tool_name"] == "lookup"
    assert structured["error"]["message"] == "failed"
    assert result_payload["content"][0]["type"] == "text"


def test_mcp_server_serve_stdio_uses_default_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeServer:
        def __init__(self, *, runtime: object | None = None) -> None:
            captured["runtime"] = runtime

        def serve(self, *, stdin: object, stdout: object) -> None:
            captured["stdin"] = stdin
            captured["stdout"] = stdout

    runtime = object()
    monkeypatch.setattr(mcp_server, "StdioMcpServer", _FakeServer)
    mcp_server._serve_stdio(runtime=runtime)

    assert captured["runtime"] is runtime
    assert captured["stdin"] is mcp_server.sys.stdin
    assert captured["stdout"] is mcp_server.sys.stdout


def test_mcp_server_cli_main_calls_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(mcp_cli, "_serve_stdio", lambda: called.append(True))

    assert mcp_cli.main() == 0
    assert called == [True]


def test_mcp_server_dunder_main_invokes_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(mcp_server, "_serve_stdio", lambda: called.append(True))

    runpy.run_module("design_research_agents.mcp_server.__main__", run_name="__main__")

    assert called == [True]
