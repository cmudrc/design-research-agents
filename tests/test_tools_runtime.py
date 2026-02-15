from __future__ import annotations

from pathlib import Path

import pytest

from design_research_agents.tools import BaseToolRuntime, ToolRuntimeConfig
from design_research_agents.tools.config import CoreToolsConfig


def _bashkit_available() -> bool:
    try:
        import bashkit  # noqa: F401
    except Exception:
        return False
    return True


def test_unified_runtime_lists_expected_core_tools() -> None:
    runtime = BaseToolRuntime()
    names = {spec.name for spec in runtime.list_tools()}

    assert "calculator_tool" in names
    assert "text_stats_tool" in names
    assert "bash.exec" in names
    assert "run.command" in names
    assert "fs.read_text" in names


def test_calculator_tool_alias_invocation() -> None:
    runtime = BaseToolRuntime()

    result = runtime.invoke(
        "calculator_tool",
        {"expression": "6 * 7"},
        request_id="unit-test",
        dependencies={},
    )

    assert result.ok is True
    assert result.output["result"] == 42.0


def test_fs_write_is_restricted_to_artifacts_by_default(tmp_path: Path) -> None:
    runtime = BaseToolRuntime(
        config=ToolRuntimeConfig(
            core_tools=CoreToolsConfig(workspace_root=str(tmp_path), artifacts_dir="artifacts")
        )
    )

    blocked = runtime.invoke(
        "fs.write_text",
        {"path": "outside.txt", "content": "nope"},
        request_id="unit-test",
        dependencies={},
    )
    assert blocked.ok is False
    assert blocked.error is not None
    assert "artifacts" in blocked.error.message

    allowed = runtime.invoke(
        "fs.write_text",
        {"path": "artifacts/ok.txt", "content": "yes", "overwrite": True},
        request_id="unit-test",
        dependencies={},
    )
    assert allowed.ok is True
    assert (tmp_path / "artifacts" / "ok.txt").read_text(encoding="utf-8") == "yes"


def test_run_command_enforces_allowlist(tmp_path: Path) -> None:
    runtime = BaseToolRuntime(
        config=ToolRuntimeConfig(
            core_tools=CoreToolsConfig(
                workspace_root=str(tmp_path),
                allowed_commands=("git",),
            )
        )
    )

    result = runtime.invoke(
        "run.command",
        {"argv": ["python3", "-c", "print('hello')"]},
        request_id="unit-test",
        dependencies={},
    )

    assert result.ok is False
    assert result.error is not None
    assert "allowed_commands" in result.error.message


@pytest.mark.skipif(
    not _bashkit_available(),
    reason="bashkit dependency is not available in this environment",
)
def test_bash_exec_returns_structured_result_and_stays_sandboxed(tmp_path: Path) -> None:
    runtime = BaseToolRuntime(
        config=ToolRuntimeConfig(core_tools=CoreToolsConfig(workspace_root=str(tmp_path)))
    )

    unique_host_path = tmp_path / "host_should_not_exist.txt"
    if unique_host_path.exists():
        unique_host_path.unlink()

    success_result = runtime.invoke(
        "bash.exec",
        {"script": "echo 'hello from bashkit'"},
        request_id="unit-test",
        dependencies={},
    )
    assert success_result.ok is True
    success_payload = success_result.output
    assert success_payload["success"] is True
    assert "hello from bashkit" in str(success_payload["stdout"])

    sandbox_result = runtime.invoke(
        "bash.exec",
        {"script": f"echo 'virtual write' > {unique_host_path}"},
        request_id="unit-test",
        dependencies={},
    )

    assert sandbox_result.ok is True
    payload = sandbox_result.output
    assert "success" in payload
    assert unique_host_path.exists() is False
