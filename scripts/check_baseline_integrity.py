"""Validate that baseline entries reference files that still exist."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

BASELINE_LINE_PATTERN = re.compile(r"^(?P<path>.+?):(?P<line>\d+):\s+DGS\d+\s+.+$")


@dataclass(slots=True, frozen=True)
class MissingBaselinePath:
    """One baseline entry whose referenced file no longer exists."""

    baseline_line: str
    """Raw baseline entry line."""
    referenced_path: str
    """Repository-relative path parsed from the baseline line."""


def _load_entries(baseline_path: Path) -> list[str]:
    """Load non-comment baseline lines.

    Args:
        baseline_path: Path to the baseline file.

    Returns:
        Baseline entries with comments/blank lines removed.
    """
    lines: list[str] = []
    for raw in baseline_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _normalize_path(path_text: str) -> str:
    """Normalize a baseline path to repository-relative POSIX format.

    Args:
        path_text: Raw path parsed from one baseline line.

    Returns:
        Normalized path text.
    """
    normalized = Path(path_text).as_posix()
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _find_missing_paths(repo_root: Path, entries: list[str]) -> list[MissingBaselinePath]:
    """Find baseline entries that point to missing files.

    Args:
        repo_root: Repository root path.
        entries: Parsed baseline entries.

    Returns:
        Missing-file violations.
    """
    missing: list[MissingBaselinePath] = []
    for entry in entries:
        match = BASELINE_LINE_PATTERN.match(entry)
        if match is None:
            continue
        relative_path = _normalize_path(match.group("path"))
        if (repo_root / relative_path).exists():
            continue
        missing.append(MissingBaselinePath(baseline_line=entry, referenced_path=relative_path))
    return missing


def main() -> int:
    """Run baseline integrity checks.

    Returns:
        ``0`` when all baseline paths exist, otherwise ``1``.

    Raises:
        FileNotFoundError: If the baseline file is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--baseline",
        default="scripts/google_docstrings_baseline.txt",
        help="Path to docstring baseline file.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    entries = _load_entries(baseline_path)
    missing = _find_missing_paths(repo_root=repo_root, entries=entries)
    if not missing:
        print("Baseline integrity checks passed.")
        return 0

    print("Baseline integrity violations detected:")
    for violation in missing:
        print(f"- missing path: {violation.referenced_path}")
        print(f"  baseline entry: {violation.baseline_line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
