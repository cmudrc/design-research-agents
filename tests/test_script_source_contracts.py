from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from design_research_agents.tools._config import ScriptTool
from design_research_agents.tools._policy import ToolPolicy, ToolPolicyConfig
from design_research_agents.tools._sources._script_source import ScriptToolSource, _run_script_tool

pytestmark = pytest.mark.contract


def _policy(tmp_path: Path, *, allow_network: bool = False) -> ToolPolicy:
    return ToolPolicy(
        ToolPolicyConfig(
            workspace_root=str(tmp_path),
            allow_network=allow_network,
        )
    )


def _script_tool(tmp_path: Path, *, path: str, **overrides: object) -> ScriptTool:
    base = {
        "name": "tool",
        "path": path,
        "description": "tool",
        "input_schema": {"type": "object", "additionalProperties": True},
        "output_schema": {"type": "object"},
        "filesystem_read": False,
        "filesystem_write": True,
        "network": False,
        "commands": (),
        "timeout_s": 3,
    }
    base.update(overrides)
    return ScriptTool(**base)


def test_script_tool_source_unknown_tool_returns_error(tmp_path: Path) -> None:
    script = tmp_path / "tool.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    source = ScriptToolSource(
        script_tools=(_script_tool(tmp_path, path=str(script)),),
        policy=_policy(tmp_path),
    )

    result = source.invoke("script::missing", {}, request_id="req", dependencies={})
    assert result.ok is False
    assert "Unknown script tool" in str(result.error)


def test_run_script_tool_rejects_network_when_policy_disabled(tmp_path: Path) -> None:
    script = tmp_path / "tool.py"
    script.write_text("print('{}')\n", encoding="utf-8")

    result = _run_script_tool(
        tool_name="script::tool",
        config=_script_tool(tmp_path, path=str(script), network=True),
        input_dict={},
        policy=_policy(tmp_path, allow_network=False),
    )
    assert result.ok is False
    assert "requires network access" in str(result.error)


def test_run_script_tool_rejects_disallowed_command_and_extension(tmp_path: Path) -> None:
    bad_ext = tmp_path / "tool.txt"
    bad_ext.write_text("noop\n", encoding="utf-8")

    unsupported = _run_script_tool(
        tool_name="script::tool",
        config=_script_tool(tmp_path, path=str(bad_ext)),
        input_dict={},
        policy=_policy(tmp_path),
    )
    assert unsupported.ok is False
    assert "Unsupported script extension" in str(unsupported.error)

    py_script = tmp_path / "tool.py"
    py_script.write_text("print('{}')\n", encoding="utf-8")
    blocked_command = _run_script_tool(
        tool_name="script::tool",
        config=_script_tool(tmp_path, path=str(py_script), commands=("curl",)),
        input_dict={},
        policy=_policy(tmp_path),
    )
    assert blocked_command.ok is False
    assert "allowed_commands" in str(blocked_command.error)


def test_run_script_tool_timeout_and_invalid_envelope_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    py_script = tmp_path / "tool.py"
    py_script.write_text("print('{}')\n", encoding="utf-8")

    def _timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd=["python3"], timeout=1, output="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", _timeout)
    timed_out = _run_script_tool(
        tool_name="script::tool",
        config=_script_tool(tmp_path, path=str(py_script), timeout_s=1),
        input_dict={},
        policy=_policy(tmp_path),
    )
    assert timed_out.ok is False
    assert isinstance(timed_out.error, type(timed_out.error))
    assert "timed out" in str(timed_out.error).lower()

    def _invalid(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="not-json", stderr="boom", returncode=1)

    monkeypatch.setattr(subprocess, "run", _invalid)
    invalid = _run_script_tool(
        tool_name="script::tool",
        config=_script_tool(tmp_path, path=str(py_script)),
        input_dict={},
        policy=_policy(tmp_path),
    )
    assert invalid.ok is False
    assert "ScriptOutputError" in str(invalid.error)


def test_run_script_tool_rejects_artifacts_outside_artifacts_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    py_script = tmp_path / "tool.py"
    py_script.write_text("print('{}')\n", encoding="utf-8")

    payload = {
        "ok": True,
        "result": {"v": 1},
        "artifacts": [{"path": "outside.txt", "mime": "text/plain"}],
        "warnings": [],
        "error": None,
    }

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(payload),
            stderr="",
            returncode=0,
        ),
    )

    result = _run_script_tool(
        tool_name="script::tool",
        config=_script_tool(tmp_path, path=str(py_script)),
        input_dict={},
        policy=_policy(tmp_path),
    )
    assert result.ok is False
    assert "outside artifacts root" in str(result.error)


def test_run_script_tool_success_path_parses_envelope(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "tool.txt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("ok\n", encoding="utf-8")

    py_script = tmp_path / "tool.py"
    py_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                "_ = json.loads(sys.stdin.read() or '{}')",
                "payload = {",
                "    'ok': True,",
                "    'result': {'value': 4},",
                "    'artifacts': [{'path': 'artifacts/tool.txt', 'mime': 'text/plain'}],",
                "    'warnings': ['warn'],",
                "    'error': None,",
                "}",
                "print(json.dumps(payload))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_script_tool(
        tool_name="script::tool",
        config=_script_tool(tmp_path, path=str(py_script)),
        input_dict={"x": 1},
        policy=_policy(tmp_path),
    )
    assert result.ok is True
    assert result.result == {"value": 4}
    assert len(result.artifacts) == 1
