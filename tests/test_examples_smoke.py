from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MONKEYPATCH_ROOT = REPO_ROOT / "tests" / "example_monkeypatch"

pytestmark = pytest.mark.examples_smoke


def _example_env(example_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env["DRA_EXAMPLE_LLM_MODE"] = "deterministic"
    env["DRA_EXAMPLE_ID"] = example_id
    env["DRA_EXAMPLE_MCP_COMMAND"] = f"{sys.executable} -m design_research_agents._mcp_server"
    existing_pythonpath = env.get("PYTHONPATH")
    test_paths = f"{EXAMPLE_MONKEYPATCH_ROOT}{os.pathsep}src"
    env["PYTHONPATH"] = f"{test_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else test_paths
    return env


def test_agent_example_smoke_runs() -> None:
    example_path = REPO_ROOT / "examples" / "agents" / "direct_llm_call.py"
    completed = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=REPO_ROOT,
        env=_example_env("examples/agents/direct_llm_call.py"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()


def test_workflow_example_smoke_runs() -> None:
    example_path = REPO_ROOT / "examples" / "workflow" / "workflow_runtime.py"
    completed = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=REPO_ROOT,
        env=_example_env("examples/workflow/workflow_runtime.py"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = ast.literal_eval(completed.stdout)
    assert payload["success"] is True


def test_script_example_smoke_runs() -> None:
    example_path = REPO_ROOT / "examples" / "tools" / "script_tools" / "bash" / "repo_quickscan.sh"
    completed = subprocess.run(
        ["bash", str(example_path)],
        cwd=REPO_ROOT,
        env=_example_env("examples/tools/script_tools/bash/repo_quickscan.sh"),
        input='{"include_hidden":false}',
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert "line_count" in payload["result"]
