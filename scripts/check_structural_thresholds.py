"""Enforce repository structural thresholds for Python module sizes."""

from __future__ import annotations

import argparse
import ast
import io
from dataclasses import dataclass
from pathlib import Path
from tokenize import COMMENT, generate_tokens

IMPLEMENTATION_SEGMENTS = {
    ("src", "design_research_agents", "implementations"),
    ("src", "design_research_agents", "_implementations"),
    ("src", "design_research_agents", "_runtime"),
    ("src", "design_research_agents", "agent", "implementations"),
    ("src", "design_research_agents", "workflow", "implementations"),
}
SCAN_ROOTS = ("src", "tests", "examples", "scripts", "docs")
IMPLEMENTATION_THRESHOLD = 900
TEST_THRESHOLD = 700
GENERAL_THRESHOLD = 700


@dataclass(slots=True, frozen=True)
class Violation:
    """One module-size threshold violation."""

    path: str
    """Repository-relative path of the violating module."""
    line_count: int
    """Effective structural line count for the module."""
    threshold: int
    """Maximum allowed effective lines for this module category."""
    category: str
    """Threshold category applied during evaluation."""


def _docstring_line_numbers(tree: ast.AST) -> set[int]:
    """Collect source lines occupied by standalone string-expression doc blocks.

    Args:
        tree: Parsed module AST.

    Returns:
        One-based line numbers for docstring-like expression blocks.
    """
    line_numbers: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr):
            continue
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        if node.end_lineno is None:
            line_numbers.add(node.lineno)
            continue
        line_numbers.update(range(node.lineno, node.end_lineno + 1))
    return line_numbers


def _comment_line_numbers(source: str) -> set[int]:
    """Collect source lines that contain Python comment tokens.

    Args:
        source: Source text to tokenize.

    Returns:
        One-based line numbers containing comment tokens.
    """
    line_numbers: set[int] = set()
    token_stream = generate_tokens(io.StringIO(source).readline)
    for token_info in token_stream:
        if token_info.type != COMMENT:
            continue
        line_numbers.add(token_info.start[0])
    return line_numbers


def _effective_line_count(path: Path) -> int:
    """Count effective structural lines excluding blank/comment/docstring-only lines.

    Args:
        path: Python file path to inspect.

    Returns:
        Effective line count used by structural threshold checks.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path.as_posix())
    docstring_lines = _docstring_line_numbers(tree)
    comment_lines = _comment_line_numbers(source)

    count = 0
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if line_number in docstring_lines:
            continue
        if line_number in comment_lines and stripped.startswith("#"):
            continue
        count += 1
    return count


def _classify(path: Path) -> tuple[int, str]:
    """Classify a file into threshold category.

    Args:
        path: Repository-relative file path.

    Returns:
        Tuple of ``(line_threshold, category_name)``.
    """
    parts = path.as_posix().split("/")
    for segment in IMPLEMENTATION_SEGMENTS:
        if tuple(parts[: len(segment)]) == segment:
            return IMPLEMENTATION_THRESHOLD, "implementation"
    if parts and parts[0] == "tests":
        return TEST_THRESHOLD, "tests"
    return GENERAL_THRESHOLD, "general"


def _collect_python_files(repo_root: Path) -> list[Path]:
    """Collect Python files under configured scan roots.

    Args:
        repo_root: Repository root directory.

    Returns:
        Sorted Python file paths.
    """
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
    """Run structural threshold checks and return process status code.

    Returns:
        ``0`` when checks pass, otherwise ``1``.
    """
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
        line_count = _effective_line_count(file_path)
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
            f"- {violation.path}: {violation.line_count} effective lines "
            f"(threshold {violation.threshold}, category={violation.category})"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
