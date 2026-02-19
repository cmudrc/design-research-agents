"""Fail when tracked junk artifacts are committed."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

BLOCKED_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|/)\.DS_Store$"), ".DS_Store"),
    (re.compile(r"(?:^|/)__pycache__(?:/|$)"), "__pycache__"),
)


def _repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Repository root directory.
    """
    return Path(__file__).resolve().parents[1]


def _tracked_paths(repo_root: Path) -> list[str]:
    """Return tracked repository paths from the git index.

    Args:
        repo_root: Repository root directory.

    Returns:
        Sorted list of tracked file paths.

    Raises:
        RuntimeError: Raised when ``git ls-files`` fails.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {stderr or 'unknown git error'}")

    output = result.stdout.decode("utf-8", errors="replace")
    return sorted(path for path in output.split("\0") if path)


def main() -> int:
    """Run tracked junk checks and return a process status code.

    Returns:
        ``0`` when checks pass, otherwise ``1``.
    """
    repo_root = _repo_root()
    offending_paths: list[str] = []
    for path in _tracked_paths(repo_root):
        for pattern, _label in BLOCKED_PATH_PATTERNS:
            if pattern.search(path):
                offending_paths.append(path)
                break

    if not offending_paths:
        print("Tracked junk check passed.")
        return 0

    print("Tracked junk artifacts detected:")
    for path in offending_paths:
        print(f"- {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
