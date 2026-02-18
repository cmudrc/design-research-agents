"""Fail when removed legacy/fallback code paths reappear in source modules."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

SCAN_ROOT = Path("src/design_research_agents")


@dataclass(slots=True, frozen=True)
class Rule:
    """One banned pattern rule used by the legacy-path checker."""

    pattern: re.Pattern[str]
    """Compiled regular-expression pattern to search for."""
    message: str
    """Human-readable violation message for matched lines."""


@dataclass(slots=True, frozen=True)
class Violation:
    """One banned-pattern match found in source code."""

    path: str
    """Repository-relative path containing the violation."""
    line: int
    """One-based line number containing the violation."""
    message: str
    """Human-readable violation description."""

    def format(self) -> str:
        """Render one deterministic violation line for CLI output.

        Returns:
            Formatted violation text.
        """
        return f"{self.path}:{self.line}: {self.message}"


def _compile_rules() -> tuple[Rule, ...]:
    """Build the banned-pattern rule set.

    Returns:
        Tuple of compiled checker rules.
    """
    return (
        Rule(
            pattern=re.compile(r'source\s*=\s*"model_legacy"'),
            message="Remove legacy controller source labels; use strict model contract only.",
        ),
        Rule(
            pattern=re.compile(r'source\s*=\s*"text_fallback"'),
            message="Remove text-prefix fallback controller branch.",
        ),
        Rule(
            pattern=re.compile(r"\bfallback_should_continue\b"),
            message="Remove continuation heuristic fallback path.",
        ),
        Rule(
            pattern=re.compile(r"\bfallback_select_tool_choice\b"),
            message="Remove JSON tool-choice fallback selector path.",
        ),
        Rule(
            pattern=re.compile(r"\bselected_alternative_index\b"),
            message="Remove legacy router output/index aliases.",
        ),
        Rule(
            pattern=re.compile(r"legacy [`]{1,2}selection[`]{1,2}"),
            message="Remove legacy router `selection` references.",
        ),
        Rule(
            pattern=re.compile(r'parsed\.get\("selection"'),
            message="Router parsing must not consume legacy `selection` field.",
        ),
        Rule(
            pattern=re.compile(r'parsed\.get\("selected_alternative_index"'),
            message="Router parsing must not consume selected_alternative_index alias.",
        ),
        Rule(
            pattern=re.compile(r'parsed\.get\("tool_name",\s*parsed\.get\("name"\)\)'),
            message="Router parsing must not consume tool_name/name aliases.",
        ),
        Rule(
            pattern=re.compile(r'raw_continue\s*=\s*parsed\.get\("continue"\)'),
            message="Direct multi-step controller must not accept continue/thought legacy schema.",
        ),
        Rule(
            pattern=re.compile(r"# noqa:\s*C901"),
            message="Disallow # noqa: C901 in src; split functions instead.",
        ),
    )


def _collect_python_files(repo_root: Path) -> list[Path]:
    """Collect Python files under the design_research_agents source tree.

    Args:
        repo_root: Repository root path.

    Returns:
        Sorted Python source files under ``src/design_research_agents``.
    """
    scan_root = repo_root / SCAN_ROOT
    if not scan_root.exists():
        return []
    return sorted(path for path in scan_root.rglob("*.py") if "__pycache__" not in path.parts)


def _scan_file(path: Path, repo_root: Path, rules: tuple[Rule, ...]) -> list[Violation]:
    """Scan one Python file for banned patterns.

    Args:
        path: Absolute Python file path to inspect.
        repo_root: Repository root path.
        rules: Compiled banned-pattern rules.

    Returns:
        Violations found in the file.
    """
    relative = path.relative_to(repo_root).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    violations: list[Violation] = []
    for line_number, line in enumerate(lines, start=1):
        for rule in rules:
            if rule.pattern.search(line) is None:
                continue
            violations.append(Violation(path=relative, line=line_number, message=rule.message))
    return violations


def main() -> int:
    """Run legacy-path checks and return process status code.

    Returns:
        ``0`` when no banned patterns are found, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current directory).",
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    rules = _compile_rules()
    violations: list[Violation] = []
    for path in _collect_python_files(repo_root):
        violations.extend(_scan_file(path, repo_root, rules))

    if not violations:
        print("No legacy/fallback/C901 path violations found.")
        return 0

    print("Legacy/fallback/C901 violations detected:")
    for violation in violations:
        print(f"- {violation.format()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
