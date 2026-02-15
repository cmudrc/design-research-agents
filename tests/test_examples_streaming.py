from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMING_EXAMPLES = (
    "direct_llm_agent_stream.py",
    "router_agent_stream.py",
    "tool_calling_agent_stream.py",
    "single_step_code_agent_stream.py",
    "multi_step_agent_stream.py",
)


@pytest.mark.parametrize("example_name", STREAMING_EXAMPLES)
def test_streaming_example_runs(example_name: str) -> None:
    example_path = REPO_ROOT / "examples" / "agents" / "streaming" / example_name
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"src{os.pathsep}{existing_pythonpath}" if existing_pythonpath else "src"

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
