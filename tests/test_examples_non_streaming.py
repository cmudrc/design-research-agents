from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
EXAMPLE_MONKEYPATCH_ROOT = REPO_ROOT / "tests" / "example_monkeypatch"
pytestmark = pytest.mark.examples_full


def _discover_non_streaming_examples() -> tuple[str, ...]:
    collected: list[str] = []
    for path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        parts = relative.parts
        if "__pycache__" in parts:
            continue
        if "streaming" in parts:
            continue
        if path.name.startswith("_"):
            continue
        collected.append(str(relative))
    return tuple(collected)


NON_STREAMING_EXAMPLES = _discover_non_streaming_examples()


@pytest.mark.parametrize("example_relpath", NON_STREAMING_EXAMPLES)
def test_non_streaming_example_runs(
    example_relpath: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRA_EXAMPLE_LLM_MODE", "deterministic")
    example_path = REPO_ROOT / example_relpath
    env = dict(os.environ)
    env["DRA_EXAMPLE_ID"] = example_relpath.replace("\\", "/")
    existing_pythonpath = env.get("PYTHONPATH")
    test_paths = f"{EXAMPLE_MONKEYPATCH_ROOT}{os.pathsep}src"
    env["PYTHONPATH"] = f"{test_paths}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else test_paths

    completed = subprocess.run(
        [sys.executable, str(example_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{example_relpath} failed with exit code {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip(), f"{example_relpath} produced empty stdout."
