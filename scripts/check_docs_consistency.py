"""Validate documentation naming and path consistency invariants."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

LEGACY_API_NAMES = (
    "PlanExecuteWorkflow",
    "ProposeAndCritiqueWorkflow",
    "AgentRoutingWorkflow",
    "PureToolWorkflow",
    "MixedAgentWorkflow",
)

LEGACY_EXAMPLE_PATHS = (
    "examples/workflow/pure_tool_workflow.py",
    "examples/workflow/mixed_agent_workflow.py",
)

SCAN_FILE_SUFFIXES = (".rst", ".md")
EXAMPLE_PATH_PATTERN = re.compile(r"(examples/[A-Za-z0-9_./-]+\.(?:py|md))")
CODE_SYMBOL_PATTERN = re.compile(r"``([A-Za-z_][A-Za-z0-9_]*)``")


@dataclass(slots=True, frozen=True)
class Violation:
    """One docs consistency violation."""

    category: str
    """Violation category identifier."""
    detail: str
    """Human-readable violation description."""


def _repo_root() -> Path:
    """Resolve repository root path.

    Returns:
        Repository root directory.
    """
    return Path(__file__).resolve().parents[1]


def _scan_files(repo_root: Path) -> list[Path]:
    """Collect documentation files to scan.

    Args:
        repo_root: Repository root directory.

    Returns:
        Sorted unique documentation and readme file paths.
    """
    files = [repo_root / "README.md"]
    docs_root = repo_root / "docs"
    if docs_root.exists():
        files.extend(
            path
            for path in sorted(docs_root.rglob("*"))
            if path.is_file()
            and path.suffix in SCAN_FILE_SUFFIXES
            and "/_build/" not in path.as_posix()
        )
    examples_root = repo_root / "examples"
    if examples_root.exists():
        files.extend(path for path in sorted(examples_root.rglob("README.md")) if path.is_file())
    return sorted(set(files))


def _find_legacy_name_violations(repo_root: Path, files: list[Path]) -> list[Violation]:
    """Find references to removed APIs and example file names.

    Args:
        repo_root: Repository root directory.
        files: Candidate files to scan.

    Returns:
        Violations for stale API names or example paths.
    """
    violations: list[Violation] = []
    patterns = {term: re.compile(rf"\b{re.escape(term)}\b") for term in LEGACY_API_NAMES}
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(repo_root).as_posix()
        for term in LEGACY_API_NAMES:
            if patterns[term].search(text):
                violations.append(
                    Violation(
                        category="legacy-name",
                        detail=f"{rel}: found removed API name '{term}'.",
                    )
                )
        for term in LEGACY_EXAMPLE_PATHS:
            if term in text:
                violations.append(
                    Violation(
                        category="legacy-example-path",
                        detail=f"{rel}: found removed example path '{term}'.",
                    )
                )
    return violations


def _find_missing_example_path_violations(repo_root: Path, files: list[Path]) -> list[Violation]:
    """Find example links that point to non-existent files.

    Args:
        repo_root: Repository root directory.
        files: Candidate files to scan.

    Returns:
        Violations for unresolved local example paths.
    """
    violations: list[Violation] = []
    referenced_paths: set[str] = set()
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in EXAMPLE_PATH_PATTERN.finditer(text):
            referenced_paths.add(match.group(1))

    for path_str in sorted(referenced_paths):
        if not (repo_root / path_str).exists():
            violations.append(
                Violation(
                    category="missing-example-path",
                    detail=f"Referenced example path does not exist: '{path_str}'.",
                )
            )
    return violations


def _parse_exports(repo_root: Path) -> set[str]:
    """Parse canonical top-level exports from package ``__init__``.

    Args:
        repo_root: Repository root directory.

    Returns:
        Exported symbol names.
    """
    init_path = repo_root / "src" / "design_research_agents" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    exports: set[str] = {"__version__"}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        target = node.target
        if not isinstance(target, ast.Name) or target.id != "_EXPORTS":
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                exports.add(key.value)
    return exports


def _parse_api_doc_symbols(repo_root: Path) -> set[str]:
    """Parse documented inline-code symbols from ``docs/api.rst``.

    Args:
        repo_root: Repository root directory.

    Returns:
        Symbol names referenced in inline code spans.
    """
    api_path = repo_root / "docs" / "api.rst"
    text = api_path.read_text(encoding="utf-8")
    return {match.group(1) for match in CODE_SYMBOL_PATTERN.finditer(text)}


def _find_export_mismatch_violations(repo_root: Path) -> list[Violation]:
    """Find mismatches between package exports and API docs.

    Args:
        repo_root: Repository root directory.

    Returns:
        Violations for missing or extra export documentation.
    """
    violations: list[Violation] = []
    exports = _parse_exports(repo_root)
    api_symbols = _parse_api_doc_symbols(repo_root)
    api_export_symbols = {symbol for symbol in api_symbols if symbol in exports}

    missing = sorted(exports - api_export_symbols)
    if missing:
        violations.append(
            Violation(
                category="api-doc-missing-export",
                detail="docs/api.rst is missing canonical exports: " + ", ".join(missing),
            )
        )

    extra = sorted(api_export_symbols - exports)
    if extra:
        violations.append(
            Violation(
                category="api-doc-extra-export",
                detail="docs/api.rst documents non-canonical exports: " + ", ".join(extra),
            )
        )
    return violations


def main() -> int:
    """Run docs consistency checks and return process status.

    Returns:
        ``0`` when checks pass, otherwise ``1``.
    """
    repo_root = _repo_root()
    files = _scan_files(repo_root)
    violations: list[Violation] = []
    violations.extend(_find_legacy_name_violations(repo_root, files))
    violations.extend(_find_missing_example_path_violations(repo_root, files))
    violations.extend(_find_export_mismatch_violations(repo_root))

    if not violations:
        print("Documentation consistency checks passed.")
        return 0

    print("Documentation consistency violations detected:")
    for violation in sorted(violations, key=lambda item: (item.category, item.detail)):
        print(f"- [{violation.category}] {violation.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
