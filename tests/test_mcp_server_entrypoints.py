from __future__ import annotations

import importlib.util
import runpy
from collections.abc import Mapping

import pytest

from design_research_agents._contracts._tools import ToolResult, ToolSpec
from design_research_agents._mcp_server import _adapters as adapters
from design_research_agents._mcp_server import _cli as mcp_cli
from design_research_agents._mcp_server import _server as mcp_server

_HAS_MCP = importlib.util.find_spec("mcp") is not None


class _RuntimeStub:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict[str, object], str]] = []

    def list_tools(self) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec(
                name="text.word_count",
                description="Count words in text.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object"},
            ),
        )

    def invoke(
        self,
        tool_name: str,
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del dependencies
        payload = dict(input)
        self.invocations.append((tool_name, payload, request_id))
        return ToolResult(tool_name=tool_name, ok=True, result={"echo": payload})


@pytest.mark.skipif(not _HAS_MCP, reason="mcp SDK unavailable")
def test_mcp_server_builds_fastmcp_tools() -> None:
    import anyio

    runtime = _RuntimeStub()
    server = mcp_server.create_mcp_server(runtime=runtime)

    tools = anyio.run(server.list_tools)
    tool = next(item for item in tools if item.name == "text.word_count")
    assert tool.inputSchema["properties"]["text"]["type"] == "string"

    _content, structured = anyio.run(server.call_tool, "text.word_count", {"text": "one two"})
    assert structured["result"] == {"echo": {"text": "one two"}}
    assert runtime.invocations == [("text.word_count", {"text": "one two"}, "mcp")]


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


def test_mcp_server_serve_stdio_uses_default_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeServer:
        def __init__(self, *, runtime: object | None = None) -> None:
            captured["runtime"] = runtime

        def run(self) -> None:
            captured["ran"] = True

    runtime = object()
    monkeypatch.setattr(mcp_server, "StdioMcpServer", _FakeServer)
    mcp_server._serve_stdio(runtime=runtime)

    assert captured["runtime"] is runtime
    assert captured["ran"] is True


def test_mcp_server_cli_main_calls_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(mcp_cli, "_serve_stdio", lambda: called.append(True))

    assert mcp_cli.main() == 0
    assert called == [True]


def test_mcp_server_dunder_main_invokes_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(mcp_server, "_serve_stdio", lambda: called.append(True))

    runpy.run_module("design_research_agents._mcp_server.__main__", run_name="__main__")

    assert called == [True]
