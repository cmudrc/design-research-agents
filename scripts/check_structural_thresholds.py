"""Enforce repository structural thresholds for Python module sizes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

IMPLEMENTATION_SEGMENTS = {
    ("src", "design_research_agents", "agent", "implementations"),
    ("src", "design_research_agents", "workflow", "implementations"),
}
SCAN_ROOTS = ("src", "tests", "examples", "scripts", "docs")


@dataclass(slots=True, frozen=True)
class Violation:
    """One module-size threshold violation."""

    path: str
    line_count: int
    threshold: int
    category: str


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _classify(path: Path) -> tuple[int, str]:
    parts = path.as_posix().split("/")
    for segment in IMPLEMENTATION_SEGMENTS:
        if tuple(parts[: len(segment)]) == segment:
            return 500, "implementation"
    if parts and parts[0] == "tests":
        return 500, "tests"
    return 650, "general"


def _collect_python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for candidate in root.rglob("*.py"):
            if "/__pycache__/" in candidate.as_posix():
                continue
            files.append(candidate)
    return sorted(files)


def main() -> int:
    """Run structural threshold checks and return process status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current directory).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    violations: list[Violation] = []

    for file_path in _collect_python_files(repo_root):
        relative_path = file_path.relative_to(repo_root)
        threshold, category = _classify(relative_path)
        line_count = _line_count(file_path)
        if line_count <= threshold:
            continue
        violations.append(
            Violation(
                path=relative_path.as_posix(),
                line_count=line_count,
                threshold=threshold,
                category=category,
            )
        )

    if not violations:
        print("Structural thresholds passed.")
        return 0

    print("Structural threshold violations detected:")
    for violation in violations:
        print(
            f"- {violation.path}: {violation.line_count} lines "
            f"(threshold {violation.threshold}, category={violation.category})"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
