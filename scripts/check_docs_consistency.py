"""Validate public documentation consistency invariants."""

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
API_AUTODOC_DIRECTIVE_PATTERN = re.compile(
    r"^\.\.\s+auto(?:class|data|function|attribute|exception)::\s+"
    r"design_research_agents\.([A-Za-z_][A-Za-z0-9_]*)\s*$",
    re.MULTILINE,
)
API_AUTOSUMMARY_ENTRY_PATTERN = re.compile(r"^design_research_agents\.([A-Za-z_][A-Za-z0-9_]*)$")
INTERNAL_MODULE_PATTERN = re.compile(
    r"\bdesign_research_agents\.(?:"
    r"implementations|_implementations|_runtime"
    r"|_[A-Za-z0-9][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*"
    r"|llm\._[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
    r"|llm\.clients\._[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
    r"|tools\._[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
    r")\b"
)
STALE_SOURCE_PATH_PATTERN = re.compile(r"\bsrc/design_research_agents/[A-Za-z0-9_./-]*")
README_BADGE_PATTERN = re.compile(r"^\[!\[(?P<alt>[^\]]+)\]\((?P<src>[^)]+)\)\]\((?P<href>[^)]+)\)$")
DOCS_HOME_BADGE_PATTERN = re.compile(
    r'<a class="drc-badge-link" href="(?P<href>[^"]+)">\s*'
    r'<img alt="(?P<alt>[^"]+)" src="(?P<src>[^"]+)">\s*'
    r"</a>",
    re.DOTALL,
)
INTERNAL_REFERENCE_DOC_PATHS = {
    "docs/reference/contracts.rst",
    "docs/reference/memory.rst",
    "docs/reference/model_selection.rst",
    "docs/reference/tracing.rst",
    "docs/reference/prompts.rst",
    "docs/reference/schemas.rst",
    "docs/reference/mcp_server.rst",
    "docs/reference/shared.rst",
}
ALLOWED_USER_DOC_INTERNAL_REFERENCES = {
    "docs/api.rst": ("design_research_agents._contracts",),
}
EXPECTED_BADGES = (
    (
        "CI",
        "https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg",
        "https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml",
    ),
    (
        "Coverage",
        "https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/coverage.svg",
        "https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml",
    ),
    (
        "Examples Passing",
        "https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-passing.svg",
        "https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml",
    ),
    (
        "Public API In Examples",
        "https://raw.githubusercontent.com/cmudrc/design-research-agents/HEAD/.github/badges/examples-api-coverage.svg",
        "https://github.com/cmudrc/design-research-agents/actions/workflows/examples.yml",
    ),
    (
        "Docs",
        "https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg",
        "https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml",
    ),
)
DOCS_AUTOMATION_REQUIRED_FILES = (
    "scripts/generate_example_docs.py",
    "scripts/generate_coverage_badge.py",
    "scripts/generate_examples_metrics.py",
    "scripts/generate_examples_badges.py",
    "scripts/update_release_readme.py",
    ".github/workflows/ci.yml",
    ".github/workflows/docs-pages.yml",
    ".github/workflows/examples.yml",
    ".github/workflows/update-release-readme.yml",
    ".github/badges/coverage.svg",
    ".github/badges/examples-passing.svg",
    ".github/badges/examples-api-coverage.svg",
    "docs/documentation_automation.rst",
)
DOCS_AUTOMATION_REQUIRED_MARKERS = {
    "docs/conf.py": (
        '"sphinx_copybutton"',
        "copybutton_prompt_text",
        "copybutton_prompt_is_regexp = True",
    ),
    "Makefile": (
        "docs-check:",
        "docs-build:",
        "docs-linkcheck:",
        "examples-smoke:",
        "examples-metrics:",
    ),
    ".github/workflows/ci.yml": (
        "make ci",
        "make docs-build",
        "make examples-metrics",
    ),
    ".github/workflows/docs-pages.yml": (
        "make docs-check",
        "make docs-build",
        "make docs-linkcheck",
    ),
    ".github/workflows/examples.yml": ("make examples-test",),
    ".github/workflows/update-release-readme.yml": ("python scripts/update_release_readme.py",),
    "docs/documentation_automation.rst": (
        "make docs-check",
        "make docs-build",
        "make docs-linkcheck",
        "examples-metrics",
        "update_release_readme.py",
        "repo-specific exception",
    ),
}
ENTRY_POINT_GUIDE_PATH = "docs/where_to_start.rst"
ENTRY_POINT_GUIDE_REQUIRED_MARKERS = (
    "``DirectLLMCall``",
    "``MultiStepAgent``",
    "``Workflow``",
    "patterns",
    "examples",
)
ENTRY_POINT_HOME_LINK_MARKER = ":doc:`where_to_start`"


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
            if path.is_file() and path.suffix in SCAN_FILE_SUFFIXES and "/_build/" not in path.as_posix()
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


def _parse_api_rendered_symbols(repo_root: Path) -> set[str]:
    """Parse exported symbols rendered by autodoc in ``docs/api.rst``.

    Args:
        repo_root: Repository root directory.

    Returns:
        Symbols rendered by ``autodata``/``autoclass``/``autosummary``.
    """
    api_path = repo_root / "docs" / "api.rst"
    text = api_path.read_text(encoding="utf-8")
    symbols = {match.group(1) for match in API_AUTODOC_DIRECTIVE_PATTERN.finditer(text)}

    in_autosummary = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == ".. autosummary::":
            in_autosummary = True
            continue
        if not in_autosummary:
            continue
        if not stripped:
            continue
        if stripped.startswith(":"):
            continue
        if not line.startswith("   "):
            in_autosummary = False
            continue
        entry_match = API_AUTOSUMMARY_ENTRY_PATTERN.fullmatch(stripped)
        if entry_match is not None:
            symbols.add(entry_match.group(1))
    return symbols


def _find_export_mismatch_violations(repo_root: Path) -> list[Violation]:
    """Find mismatches between package exports and rendered API docs.

    Args:
        repo_root: Repository root directory.

    Returns:
        Violations for missing or extra export coverage.
    """
    violations: list[Violation] = []
    exports = _parse_exports(repo_root)
    rendered_symbols = _parse_api_rendered_symbols(repo_root)

    missing = sorted(exports - rendered_symbols)
    if missing:
        violations.append(
            Violation(
                category="api-doc-missing-export",
                detail="docs/api.rst is missing rendered export coverage for: " + ", ".join(missing),
            )
        )

    extra = sorted(rendered_symbols - exports)
    if extra:
        violations.append(
            Violation(
                category="api-doc-extra-export",
                detail="docs/api.rst renders non-canonical exports: " + ", ".join(extra),
            )
        )
    return violations


def _find_internal_module_boundary_violations(
    repo_root: Path,
    files: list[Path],
) -> list[Violation]:
    """Find user-facing docs that reference internal implementation modules.

    Args:
        repo_root: Repository root directory.
        files: Candidate files to scan.

    Returns:
        Violations for internal module leakage in public docs.
    """
    violations: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(repo_root).as_posix()
        if rel in INTERNAL_REFERENCE_DOC_PATHS:
            continue
        for match in INTERNAL_MODULE_PATTERN.finditer(text):
            matched_path = match.group(0)
            allowed_prefixes = ALLOWED_USER_DOC_INTERNAL_REFERENCES.get(rel, ())
            if any(matched_path.startswith(prefix) for prefix in allowed_prefixes):
                continue
            key = (rel, matched_path)
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                Violation(
                    category="public-doc-internal-module",
                    detail=f"{rel}: references internal module path '{matched_path}'.",
                )
            )
    return violations


def _find_stale_source_path_violations(repo_root: Path, files: list[Path]) -> list[Violation]:
    """Find stale source-tree path references in user-facing docs.

    Args:
        repo_root: Repository root directory.
        files: Candidate files to scan.

    Returns:
        Violations for stale ``src/design_research_agents/...`` references.
    """
    violations: list[Violation] = []
    seen: set[tuple[str, str]] = set()
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(repo_root).as_posix()
        for match in STALE_SOURCE_PATH_PATTERN.finditer(text):
            matched_path = match.group(0)
            key = (rel, matched_path)
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                Violation(
                    category="stale-source-path",
                    detail=f"{rel}: found stale source-tree path '{matched_path}'.",
                )
            )
    return violations


def _format_badges(badges: tuple[tuple[str, str, str], ...]) -> str:
    """Render badge tuples for deterministic violation messages.

    Args:
        badges: Ordered ``(alt, src, href)`` badge tuples.

    Returns:
        Compact human-readable badge description.
    """
    return "; ".join(f"{alt} -> {src} -> {href}" for alt, src, href in badges)


def _parse_readme_badges(readme_text: str) -> tuple[tuple[str, str, str], ...]:
    """Parse the contiguous top-of-file badge block from ``README.md``.

    Args:
        readme_text: Full README markdown text.

    Returns:
        Ordered badge tuples in ``(alt, src, href)`` form.
    """
    badges: list[tuple[str, str, str]] = []
    collecting = False
    for raw_line in readme_text.splitlines()[1:]:
        stripped = raw_line.strip()
        match = README_BADGE_PATTERN.fullmatch(stripped)
        if match is not None:
            badges.append((match.group("alt"), match.group("src"), match.group("href")))
            collecting = True
            continue
        if collecting:
            break
    return tuple(badges)


def _parse_docs_home_badges(index_text: str) -> tuple[tuple[str, str, str], ...]:
    """Parse the badge row rendered on the docs home page.

    Args:
        index_text: ``docs/index.rst`` source text.

    Returns:
        Ordered badge tuples in ``(alt, src, href)`` form.
    """
    badges = [
        (match.group("alt"), match.group("src"), match.group("href"))
        for match in DOCS_HOME_BADGE_PATTERN.finditer(index_text)
    ]
    return tuple(badges)


def _find_badge_surface_violations(repo_root: Path) -> list[Violation]:
    """Validate README and docs-home badge surfaces against the shared baseline.

    Args:
        repo_root: Repository root directory.

    Returns:
        Badge-surface violations.
    """
    violations: list[Violation] = []

    readme_path = repo_root / "README.md"
    readme_badges = _parse_readme_badges(readme_path.read_text(encoding="utf-8"))
    if readme_badges != EXPECTED_BADGES:
        violations.append(
            Violation(
                category="badge-surface",
                detail=(
                    "README.md badge block does not match the shared baseline. "
                    f"expected [{_format_badges(EXPECTED_BADGES)}], got [{_format_badges(readme_badges)}]"
                ),
            )
        )

    docs_index_path = repo_root / "docs" / "index.rst"
    docs_badges = _parse_docs_home_badges(docs_index_path.read_text(encoding="utf-8"))
    if docs_badges != EXPECTED_BADGES:
        violations.append(
            Violation(
                category="badge-surface",
                detail=(
                    "docs/index.rst badge row does not match the shared baseline. "
                    f"expected [{_format_badges(EXPECTED_BADGES)}], got [{_format_badges(docs_badges)}]"
                ),
            )
        )

    return violations


def _find_docs_automation_baseline_violations(repo_root: Path) -> list[Violation]:
    """Validate the checked docs/CI automation baseline for this repository.

    Args:
        repo_root: Repository root directory.

    Returns:
        Docs-automation baseline violations.
    """
    violations: list[Violation] = []

    for relative_path in DOCS_AUTOMATION_REQUIRED_FILES:
        if not (repo_root / relative_path).exists():
            violations.append(
                Violation(
                    category="docs-automation-baseline",
                    detail=f"required automation file is missing: '{relative_path}'.",
                )
            )

    for relative_path, markers in DOCS_AUTOMATION_REQUIRED_MARKERS.items():
        path = repo_root / relative_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                violations.append(
                    Violation(
                        category="docs-automation-baseline",
                        detail=f"{relative_path}: missing required baseline marker '{marker}'.",
                    )
                )

    return violations


def _find_entry_point_guide_violations(repo_root: Path) -> list[Violation]:
    """Validate the newcomer entry-point guide and its home-page link.

    Args:
        repo_root: Repository root directory.

    Returns:
        Entry-point guide violations.
    """
    violations: list[Violation] = []

    docs_index_path = repo_root / "docs" / "index.rst"
    docs_index_text = docs_index_path.read_text(encoding="utf-8")
    if ENTRY_POINT_HOME_LINK_MARKER not in docs_index_text:
        violations.append(
            Violation(
                category="entry-point-guide",
                detail=f"docs/index.rst must link to '{ENTRY_POINT_GUIDE_PATH}'.",
            )
        )

    guide_path = repo_root / ENTRY_POINT_GUIDE_PATH
    if not guide_path.exists():
        violations.append(
            Violation(
                category="entry-point-guide",
                detail=f"missing required entry-point guide '{ENTRY_POINT_GUIDE_PATH}'.",
            )
        )
        return violations

    guide_text = guide_path.read_text(encoding="utf-8")
    for marker in ENTRY_POINT_GUIDE_REQUIRED_MARKERS:
        if marker not in guide_text:
            violations.append(
                Violation(
                    category="entry-point-guide",
                    detail=f"{ENTRY_POINT_GUIDE_PATH}: missing required marker '{marker}'.",
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
    violations.extend(_find_internal_module_boundary_violations(repo_root, files))
    violations.extend(_find_stale_source_path_violations(repo_root, files))
    violations.extend(_find_badge_surface_violations(repo_root))
    violations.extend(_find_docs_automation_baseline_violations(repo_root))
    violations.extend(_find_entry_point_guide_violations(repo_root))

    if not violations:
        print("Documentation consistency checks passed.")
        return 0

    print("Documentation consistency violations detected:")
    for violation in sorted(violations, key=lambda item: (item.category, item.detail)):
        print(f"- [{violation.category}] {violation.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
