from __future__ import annotations

import importlib.util
import runpy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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


def test_stdio_wrapper_requires_process_streams_and_forwards_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    wrapped = SimpleNamespace(run=lambda *, transport: calls.append(transport))
    monkeypatch.setattr(mcp_server, "create_mcp_server", lambda *, runtime=None: wrapped)
    server = mcp_server.StdioMcpServer(runtime=object())  # type: ignore[arg-type]

    server.run()
    server.serve(stdin=mcp_server.sys.stdin, stdout=mcp_server.sys.stdout)
    assert calls == ["stdio", "stdio"]
    with pytest.raises(RuntimeError, match="requires process stdin/stdout"):
        server.serve(stdin=object(), stdout=object())


def test_create_mcp_server_registers_runtime_tools_and_preserves_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RegisteredTool:
        def __init__(self) -> None:
            self.parameters: dict[str, object] = {}

    class _FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name
            self._tool_manager = SimpleNamespace(_tools={})

        def add_tool(self, tool: object, *, name: str, description: str, structured_output: bool) -> None:
            del tool, description, structured_output
            self._tool_manager._tools[name] = _RegisteredTool()

    runtime = _RuntimeStub()
    monkeypatch.setattr(mcp_server, "_import_fastmcp", lambda: _FakeFastMCP)
    server = mcp_server.create_mcp_server(runtime=runtime)

    assert server.name == "design-research-agents"
    assert server._tool_manager._tools["text.word_count"].parameters == runtime.list_tools()[0].input_schema


def test_runtime_tool_removes_omitted_values_and_surfaces_failures() -> None:
    import anyio

    spec = ToolSpec(
        name="demo.tool-name",
        description="Demo.",
        input_schema={
            "type": "object",
            "properties": {"required": {"type": "string"}, "optional": {"type": "integer"}},
            "required": ["required"],
        },
        output_schema={"type": "object"},
    )

    class _Runtime:
        def __init__(self, *, ok: bool) -> None:
            self.ok = ok
            self.payload: dict[str, object] | None = None

        def invoke(self, tool_name: str, input: Mapping[str, object], **_kwargs: object) -> ToolResult:
            self.payload = dict(input)
            return ToolResult(
                tool_name=tool_name,
                ok=self.ok,
                result={"path": Path("result.txt")},
                error=None if self.ok else "failed cleanly",
            )

    success_runtime = _Runtime(ok=True)
    runtime_tool = mcp_server._build_runtime_tool(runtime=success_runtime, spec=spec)  # type: ignore[arg-type]
    payload = anyio.run(lambda: runtime_tool(required="value", optional=None))
    assert runtime_tool.__name__ == "tool_demo_tool_name"
    assert success_runtime.payload == {"required": "value"}
    assert payload["result"] == {"path": "result.txt"}

    failing_tool = mcp_server._build_runtime_tool(runtime=_Runtime(ok=False), spec=spec)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="failed cleanly"):
        anyio.run(lambda: failing_tool(required="value"))


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"type": ["null", "string"]}, str),
        ({"type": "number"}, float),
        ({"type": "integer"}, int),
        ({"type": "boolean"}, bool),
        ({"type": "array"}, list[object]),
        ({"type": "object"}, dict[str, object]),
        ({"type": ["null"]}, object),
        ("invalid", object),
    ],
)
def test_mcp_schema_annotations_cover_supported_json_types(schema: object, expected: object) -> None:
    assert mcp_server._annotation_from_schema(schema) == expected


def test_mcp_signature_normalizes_names_and_rejects_collisions() -> None:
    empty = mcp_server._signature_from_schema(tool_name="scalar", input_schema={"type": "string"})
    malformed = mcp_server._signature_from_schema(
        tool_name="malformed",
        input_schema={"properties": "invalid", "required": "invalid"},
    )
    renamed = mcp_server._signature_from_schema(
        tool_name="renamed",
        input_schema={"properties": {3: {}, "not-valid": {"type": "string"}}},  # type: ignore[dict-item]
    )

    assert not empty.parameters
    assert not malformed.parameters
    assert list(renamed.parameters) == ["not_valid"]
    assert renamed.parameters["not_valid"].default is None
    with pytest.raises(ValueError, match="Cannot expose tool 'collision'"):
        mcp_server._signature_from_schema(
            tool_name="collision",
            input_schema={"properties": {"a-b": {}, "a_b": {}}},
        )


@dataclass(frozen=True)
class _NestedPayload:
    value: int


def test_mcp_json_conversion_and_safe_identifiers_cover_nested_values() -> None:
    assert mcp_server._to_jsonable(_NestedPayload(value=2)) == {"value": 2}
    assert mcp_server._to_jsonable((Path("a.txt"), b"bytes\xff", object()))[:2] == ["a.txt", "bytes�"]
    assert mcp_server._safe_identifier("") == "tool_"
    assert mcp_server._safe_identifier("2tool") == "tool_2tool"
    assert mcp_server._safe_identifier("valid_name") == "valid_name"


@pytest.mark.skipif(_HAS_MCP, reason="dependency-error path only applies without the MCP extra")
def test_mcp_import_reports_install_hint_without_optional_sdk() -> None:
    with pytest.raises(mcp_server.McpServerDependencyError, match=r"design-research-agents\[mcp\]"):
        mcp_server._import_fastmcp()
