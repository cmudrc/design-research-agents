from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from design_research_agents import cli
from design_research_agents.contracts.tools import ToolMetadata, ToolResult, ToolSpec
from design_research_agents.tools.config import McpServer, ScriptTool


class _FakeRuntime:
    def __init__(self, *, server_ids: tuple[str, ...] = ()) -> None:
        self.config = SimpleNamespace(
            mcp=SimpleNamespace(
                servers=tuple(
                    McpServer(id=server_id, command=("echo",)) for server_id in server_ids
                )
            )
        )
        self.last_tool: str | None = None
        self.closed = False

    def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="alpha::one",
                description="x",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                metadata=ToolMetadata(source="mcp"),
            ),
            ToolSpec(
                name="script::quick",
                description="x",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                metadata=ToolMetadata(source="script"),
            ),
            ToolSpec(
                name="core::skip",
                description="x",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                metadata=ToolMetadata(source="core"),
            ),
        ]

    def invoke(
        self,
        tool_name: str,
        input_dict: dict[str, object],
        *,
        request_id: str,
        dependencies: dict[str, object],
    ) -> ToolResult:
        del input_dict, request_id, dependencies
        self.last_tool = tool_name
        if tool_name.endswith("fail"):
            return ToolResult(tool_name=tool_name, ok=False, error="failed")
        return ToolResult(tool_name=tool_name, ok=True, result={"ok": True})

    def close(self) -> None:
        self.closed = True


def test_main_prints_help_when_no_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "usage: dra" in captured.out


def test_handle_mcp_serve_invokes_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(cli, "_serve_stdio", lambda: called.append(True))

    args = argparse.Namespace(mcp_command="serve")

    assert cli._handle_mcp(args) == 0
    assert called == [True]


def test_handle_mcp_ping_missing_server(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(cli, "_build_runtime", lambda _path: runtime)

    args = argparse.Namespace(mcp_command="ping", server="missing", config=None)
    exit_code = cli._handle_mcp(args)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "is not configured" in captured.out
    assert runtime.closed is True


def test_handle_mcp_ping_and_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _FakeRuntime(server_ids=("alpha",))
    monkeypatch.setattr(cli, "_build_runtime", lambda _path: runtime)

    ping_args = argparse.Namespace(mcp_command="ping", server="alpha", config=None)
    assert cli._handle_mcp(ping_args) == 0
    ping_payload = json.loads(capsys.readouterr().out)
    assert ping_payload == {"server": "alpha", "tools": ["alpha::one"]}

    bad_json_args = argparse.Namespace(
        mcp_command="call", tool="alpha::one", json="[]", config=None
    )
    assert cli._handle_mcp(bad_json_args) == 1
    assert "--json must be a valid JSON object" in capsys.readouterr().out

    call_args = argparse.Namespace(
        mcp_command="call",
        tool="alpha::one",
        json='{"value": 1}',
        config=None,
    )
    assert cli._handle_mcp(call_args) == 0
    assert runtime.last_tool == "alpha::one"

    call_fail_args = argparse.Namespace(
        mcp_command="call",
        tool="alpha::fail",
        json='{"value": 1}',
        config=None,
    )
    assert cli._handle_mcp(call_fail_args) == 2


def test_handle_mcp_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli._handle_mcp(argparse.Namespace(mcp_command="unknown")) == 1
    assert "Unknown mcp command" in capsys.readouterr().out


def test_handle_script_lint_list_and_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(cli, "_build_runtime", lambda _path: runtime)

    script_file = tmp_path / "tool.py"
    script_file.write_text("print('ok')\n", encoding="utf-8")
    lint_ok = argparse.Namespace(script_command="lint", target=str(tmp_path))
    assert cli._handle_script(lint_ok) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "diagnostics": []}

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    lint_bad = argparse.Namespace(script_command="lint", target=str(empty_dir))
    assert cli._handle_script(lint_bad) == 2
    lint_payload = json.loads(capsys.readouterr().out)
    assert lint_payload["ok"] is False

    list_args = argparse.Namespace(script_command="list", config=None)
    assert cli._handle_script(list_args) == 0
    assert json.loads(capsys.readouterr().out) == {"tools": ["script::quick"]}

    run_bad_json = argparse.Namespace(
        script_command="run",
        tool_name="quick",
        json="[]",
        config=None,
    )
    assert cli._handle_script(run_bad_json) == 1

    run_ok = argparse.Namespace(
        script_command="run",
        tool_name="quick",
        json='{"x": 1}',
        config=None,
    )
    assert cli._handle_script(run_ok) == 0
    assert runtime.last_tool == "script::quick"

    run_prefixed = argparse.Namespace(
        script_command="run",
        tool_name="script::quick",
        json='{"x": 1}',
        config=None,
    )
    assert cli._handle_script(run_prefixed) == 0
    assert runtime.last_tool == "script::quick"


def test_handle_script_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli._handle_script(argparse.Namespace(script_command="unknown")) == 1
    assert "Unknown script command" in capsys.readouterr().out


def test_build_runtime_parse_json_object_and_server_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RuntimeFactory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_config = SimpleNamespace(
        core_tools=SimpleNamespace(enabled=False, workspace_root="/tmp/work"),
        mcp=SimpleNamespace(enabled=True, servers=(McpServer(id="alpha", command=("echo",)),)),
        script_tools=SimpleNamespace(
            enabled=True,
            tools=(
                ScriptTool(
                    name="quick",
                    path="/tmp/quick.py",
                    description="quick",
                ),
            ),
        ),
    )

    monkeypatch.setattr(cli, "Toolbox", _RuntimeFactory)
    monkeypatch.setattr(cli, "load_tool_runtime_config", lambda _path: fake_config)

    default_runtime = cli._build_runtime(None)
    yaml_runtime = cli._build_runtime("config.yaml")

    assert isinstance(default_runtime, _RuntimeFactory)
    assert default_runtime.kwargs == {}
    assert yaml_runtime.kwargs["workspace_root"] == "/tmp/work"
    assert yaml_runtime.kwargs["enable_core_tools"] is False
    assert yaml_runtime.kwargs["mcp_servers"][0].id == "alpha"
    assert yaml_runtime.kwargs["script_tools"][0].name == "quick"

    assert cli._parse_json_object('{"a": 1}') == {"a": 1}
    assert cli._parse_json_object("not-json") is None
    assert cli._parse_json_object("[]") is None

    servers = (McpServer(id="alpha", command=("echo",)),)
    assert cli._server_exists(servers, "alpha") is True
    assert cli._server_exists(servers, "beta") is False


def test_main_parses_subcommands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_handle_mcp", lambda _args: 7)
    monkeypatch.setattr(cli, "_handle_script", lambda _args: 8)

    assert cli.main(["mcp", "serve"]) == 7
    assert cli.main(["script", "list"]) == 8
