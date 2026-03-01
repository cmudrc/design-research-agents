"""Consistency checks for contract serialization naming."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "design_research_agents"


def _collect_violations(*, pattern: re.Pattern[str], root: Path, ignore: set[Path] | None = None) -> list[str]:
    violations: list[str] = []
    ignored = ignore or set()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path in ignored:
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
    return violations


def test_contract_code_does_not_define_asdict_methods() -> None:
    pattern = re.compile(r"^\s*def\s+asdict\(")
    violations = _collect_violations(pattern=pattern, root=SRC_ROOT)
    assert violations == [], "\n".join(violations)


def test_project_code_does_not_call_non_dataclasses_asdict_methods() -> None:
    pattern = re.compile(r"(?<!dataclasses)\.asdict\(")
    violations: list[str] = []
    ignored = {Path(__file__).resolve()}
    for root in (SRC_ROOT, REPO_ROOT / "tests", REPO_ROOT / "examples"):
        violations.extend(_collect_violations(pattern=pattern, root=root, ignore=ignored))
    assert violations == [], "\n".join(sorted(violations))
