from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MONKEYPATCH_ROOT = REPO_ROOT / "tests" / "example_monkeypatch"


def test_lazy_shell_example_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRA_EXAMPLE_LLM_MODE", "deterministic")
    example_path = REPO_ROOT / "examples" / "lazy_tools" / "bash" / "repo_quickscan.sh"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    test_paths = f"{EXAMPLE_MONKEYPATCH_ROOT}{os.pathsep}src"
    env["PYTHONPATH"] = (
        f"{test_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else test_paths
    )

    completed = subprocess.run(
        ["bash", str(example_path)],
        cwd=REPO_ROOT,
        env=env,
        input='{"include_hidden":false}',
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{example_path} failed with exit code {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert isinstance(payload["result"], dict)
    assert "line_count" in payload["result"]
