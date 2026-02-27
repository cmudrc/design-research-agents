from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest

import design_research_agents.tools._registry as tool_registry
import design_research_agents.tools._sources._mcp_source as mcp_source
from design_research_agents.tools._config import McpConfig, MCPServerConfig
from design_research_agents.tools._policy import ToolPolicy, ToolPolicyConfig
from design_research_agents.tools._registry import ToolRegistry
from design_research_agents.tools._sources._mcp_source import McpProtocolError, McpToolSource

pytestmark = pytest.mark.contract


class _StubClient:
    def __init__(
        self,
        *,
        tools: list[dict[str, object]],
        call_payload: dict[str, object] | None = None,
    ):
        self._tools = tools
        self._call_payload = call_payload or {"structuredContent": {"ok": True, "result": {}}}

    def list_tools(self) -> list[dict[str, object]]:
        return list(self._tools)

    def call_tool(self, *, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        del tool_name, arguments
        return dict(self._call_payload)

    def close(self) -> None:
        pass


@dataclass(slots=True)
class _FakeProcess:
    stdout: object
    stderr: object

    def poll(self) -> int | None:
        return 0


def _policy(tmp_path: Path) -> ToolPolicy:
    return ToolPolicy(ToolPolicyConfig(workspace_root=str(tmp_path)))


def _config() -> McpConfig:
    return McpConfig(
        enabled=True,
        servers=(MCPServerConfig(id="alpha", command=("python3", "-c", "pass")),),
    )


def test_mcp_source_route_refresh_and_invoke_variants(tmp_path: Path) -> None:
    source = McpToolSource(mcp_config=_config(), policy=_policy(tmp_path))
    source._clients["alpha"] = _StubClient(
        tools=[
            {
                "name": "sum",
                "description": "sum",
                "inputSchema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "additionalProperties": True,
                },
            },
            {"name": "raw", "inputSchema": "bad"},
            {"description": "missing-name"},
        ]
    )

    specs = source.list_tools()
    names = {spec.name for spec in specs}
    assert names == {"alpha::raw", "alpha::sum"}

    unknown = source.invoke("alpha::missing", {}, request_id="req", dependencies={})
    assert unknown.ok is False
    assert "Unknown MCP tool" in str(unknown.error)

    source._refresh_routes()
    source._routes = {"alpha::sum": source._routes["alpha::sum"]}
    source._clients.pop("alpha")
    source._refresh_routes = lambda: None  # type: ignore[method-assign]
    missing_client = source.invoke("alpha::sum", {}, request_id="req", dependencies={})
    assert missing_client.ok is False
    assert "not configured" in str(missing_client.error)


def test_mcp_source_invoke_handles_structured_and_text_payloads(tmp_path: Path) -> None:
    source = McpToolSource(mcp_config=_config(), policy=_policy(tmp_path))

    source._clients["alpha"] = _StubClient(
        tools=[{"name": "sum", "inputSchema": {"type": "object"}}],
        call_payload={
            "structuredContent": {
                "ok": True,
                "result": {"value": 3},
                "artifacts": [],
                "warnings": ["w"],
                "error": None,
            }
        },
    )
    structured = source.invoke("alpha::sum", {"a": 1}, request_id="req", dependencies={})
    assert structured.ok is True
    assert structured.result == {"value": 3}

    source._clients["alpha"] = _StubClient(
        tools=[{"name": "sum", "inputSchema": {"type": "object"}}],
        call_payload={"structuredContent": {"value": 7}, "isError": False},
    )
    mapped = source.invoke("alpha::sum", {}, request_id="req", dependencies={})
    assert mapped.ok is True
    assert mapped.result == {"value": 7}

    source._clients["alpha"] = _StubClient(
        tools=[{"name": "sum", "inputSchema": {"type": "object"}}],
        call_payload={
            "isError": False,
            "content": [{"type": "text", "text": json.dumps({"answer": 42})}],
        },
    )
    parsed_json = source.invoke("alpha::sum", {}, request_id="req", dependencies={})
    assert parsed_json.result == {"answer": 42}

    source._clients["alpha"] = _StubClient(
        tools=[{"name": "sum", "inputSchema": {"type": "object"}}],
        call_payload={"isError": True, "content": [{"type": "text", "text": "plain"}]},
    )
    text_error = source.invoke("alpha::sum", {}, request_id="req", dependencies={})
    assert text_error.ok is False
    assert text_error.result == "plain"


def test_mcp_source_invoke_handles_client_exceptions(tmp_path: Path) -> None:
    source = McpToolSource(mcp_config=_config(), policy=_policy(tmp_path))

    class _RaisingClient(_StubClient):
        def call_tool(self, *, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
            del tool_name, arguments
            raise RuntimeError("call failed")

    source._clients["alpha"] = _RaisingClient(tools=[{"name": "sum", "inputSchema": {"type": "object"}}])
    result = source.invoke("alpha::sum", {}, request_id="req", dependencies={})
    assert result.ok is False
    assert "call failed" in str(result.error)


def test_mcp_source_registry_emits_observation_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = McpToolSource(mcp_config=_config(), policy=_policy(tmp_path))
    source._clients["alpha"] = _StubClient(
        tools=[{"name": "sum", "inputSchema": {"type": "object"}}],
        call_payload={"structuredContent": {"ok": True, "result": {"value": 3}}},
    )

    registry = ToolRegistry()
    registry.add_source(source)
    captured: dict[str, list[dict[str, object]]] = {"invocations": [], "results": []}
    monkeypatch.setattr(
        tool_registry,
        "emit_tool_invocation_observed",
        lambda **kwargs: captured["invocations"].append(dict(kwargs)),
    )
    monkeypatch.setattr(
        tool_registry,
        "emit_tool_result_observed",
        lambda **kwargs: captured["results"].append(dict(kwargs)),
    )

    result = registry.invoke("alpha::sum", {"a": 1}, request_id="req-observed", dependencies={"x": 1})
    assert result.ok is True
    assert captured["invocations"]
    assert captured["results"]
    assert captured["invocations"][-1]["source_id"] == "mcp"
    assert captured["results"][-1]["ok"] is True


def test_stdio_read_response_handles_timeout_and_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = ToolPolicy(ToolPolicyConfig(workspace_root="."))
    client = mcp_source._StdioMcpClient(
        server=MCPServerConfig(id="alpha", command=("python3", "-c", "pass"), timeout_s=1),
        policy=policy,
    )

    class _FakeStdout:
        def readline(self) -> str:
            return ""

    process = _FakeProcess(stdout=_FakeStdout(), stderr=StringIO("boom"))
    client._process = process  # type: ignore[assignment]

    monkeypatch.setattr(
        mcp_source.select,
        "select",
        lambda *_args, **_kwargs: ([process.stdout], [], []),
    )
    with pytest.raises(McpProtocolError, match="closed unexpectedly"):
        client._read_response(expected_id=1)

    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(mcp_source.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(mcp_source.select, "select", lambda *_args, **_kwargs: ([], [], []))
    with pytest.raises(McpProtocolError, match="Timed out waiting"):
        client._read_response(expected_id=1)


def test_stdio_read_response_skips_noise_until_expected_id(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = ToolPolicy(ToolPolicyConfig(workspace_root="."))
    client = mcp_source._StdioMcpClient(
        server=MCPServerConfig(id="alpha", command=("python3", "-c", "pass"), timeout_s=5),
        policy=policy,
    )

    class _FakeStdout:
        def __init__(self) -> None:
            self.lines = [
                "not-json\n",
                '{"jsonrpc":"2.0","id":2,"result":{}}\n',
                '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n',
            ]

        def readline(self) -> str:
            if not self.lines:
                return ""
            return self.lines.pop(0)

    process = _FakeProcess(stdout=_FakeStdout(), stderr=StringIO(""))
    client._process = process  # type: ignore[assignment]
    monkeypatch.setattr(
        mcp_source.select,
        "select",
        lambda *_args, **_kwargs: ([process.stdout], [], []),
    )

    payload = client._read_response(expected_id=1)
    assert payload["result"] == {"ok": True}
