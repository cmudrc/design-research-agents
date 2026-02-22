"""Tests for structural-threshold checker classification and enforcement."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_structural_thresholds.py"


def _write_python_lines(repo_root: Path, relative_path: str, *, line_count: int) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['"""Module summary."""']
    lines.extend(f"value_{index} = {index}" for index in range(line_count))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_checker(repo_root: Path) -> tuple[int, list[str]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--repo-root",
            str(repo_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return completed.returncode, output_lines


def test_implementation_paths_are_bucketed_as_implementation() -> None:
    repo_root = Path("artifacts/tests/structural_threshold_impl_bucket")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_python_lines(
        repo_root,
        "src/design_research_agents/implementations/patterns/example.py",
        line_count=901,
    )

    exit_code, output_lines = _run_checker(repo_root)

    assert exit_code == 1
    assert any(
        "src/design_research_agents/implementations/patterns/example.py" in line and "category=implementation" in line
        for line in output_lines
    )


def test_general_paths_are_not_misclassified_as_implementation() -> None:
    repo_root = Path("artifacts/tests/structural_threshold_general_bucket")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_python_lines(
        repo_root,
        "src/design_research_agents/workflow/core/example.py",
        line_count=901,
    )

    exit_code, output_lines = _run_checker(repo_root)

    assert exit_code == 1
    assert any(
        "src/design_research_agents/workflow/core/example.py" in line and "category=general" in line
        for line in output_lines
    )
