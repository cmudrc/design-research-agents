from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MONKEYPATCH_ROOT = REPO_ROOT / "tests" / "example_monkeypatch"
STREAMING_EXAMPLES = (
    "single_step_direct_llm_agent_stream.py",
    "single_step_tool_router_agent_stream.py",
    "single_step_router_agent_stream.py",
    "single_step_json_tool_calling_agent_stream.py",
    "single_step_code_tool_calling_agent_stream.py",
    "multi_step_code_tool_calling_agent_stream.py",
    "multi_step_json_tool_calling_agent_stream.py",
    "multi_step_tool_router_agent_stream.py",
    "multi_step_direct_llm_agent_stream.py",
)


@pytest.mark.parametrize("example_name", STREAMING_EXAMPLES)
def test_streaming_example_runs(
    example_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRA_EXAMPLE_LLM_MODE", "deterministic")
    example_path = REPO_ROOT / "examples" / "agents" / "streaming" / example_name
    env = dict(os.environ)
    env["DRA_EXAMPLE_ID"] = f"examples/agents/streaming/{example_name}"
    existing_pythonpath = env.get("PYTHONPATH")
    test_paths = f"{EXAMPLE_MONKEYPATCH_ROOT}{os.pathsep}src"
    env["PYTHONPATH"] = (
        f"{test_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else test_paths
    )

    completed = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{example_name} failed with exit code {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "delta:" in completed.stdout
    assert "completed:" in completed.stdout
    assert '"success": true' in completed.stdout
