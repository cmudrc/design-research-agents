"""Tests for the documentation consistency checker."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_docs_consistency.py"


def _load_checker_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_docs_consistency", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_file(repo_root: Path, relative_path: str, content: str) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def _seed_badge_surfaces(repo_root: Path, checker_module: ModuleType) -> None:
    readme_lines = ["# design-research-agents"]
    docs_badges: list[str] = []
    for alt, src, href in checker_module.EXPECTED_BADGES:
        readme_lines.append(f"[![{alt}]({src})]({href})")
        docs_badges.append(
            f'    <a class="drc-badge-link" href="{href}">\n      <img alt="{alt}" src="{src}">\n    </a>'
        )

    _write_file(repo_root, "README.md", "\n".join(readme_lines) + "\n")
    _write_file(
        repo_root,
        "docs/index.rst",
        f"""
        Home
        ====

        {checker_module.ENTRY_POINT_HOME_LINK_MARKER}

        .. raw:: html

           <div class="drc-badge-row">
        {"\n".join(docs_badges)}
           </div>
        """,
    )


def _seed_docs_automation_baseline(repo_root: Path, checker_module: ModuleType) -> None:
    for relative_path in checker_module.DOCS_AUTOMATION_REQUIRED_FILES:
        _write_file(repo_root, relative_path, "placeholder\n")

    for relative_path, markers in checker_module.DOCS_AUTOMATION_REQUIRED_MARKERS.items():
        _write_file(repo_root, relative_path, "\n".join(markers) + "\n")

    _write_file(
        repo_root,
        checker_module.ENTRY_POINT_GUIDE_PATH,
        "\n".join(checker_module.ENTRY_POINT_GUIDE_REQUIRED_MARKERS) + "\n",
    )


def test_badge_surface_checker_accepts_matching_readme_and_docs_home(tmp_path: Path) -> None:
    checker_module = _load_checker_module()
    _seed_badge_surfaces(tmp_path, checker_module)

    violations = checker_module._find_badge_surface_violations(tmp_path)

    assert violations == []


def test_badge_surface_checker_reports_docs_home_drift(tmp_path: Path) -> None:
    checker_module = _load_checker_module()
    _seed_badge_surfaces(tmp_path, checker_module)
    docs_index = tmp_path / "docs" / "index.rst"
    docs_index.write_text(
        docs_index.read_text(encoding="utf-8").replace("Examples Passing", "Examples Stable", 1),
        encoding="utf-8",
    )

    violations = checker_module._find_badge_surface_violations(tmp_path)

    assert [violation.category for violation in violations] == ["badge-surface"]
    assert "docs/index.rst badge row does not match" in violations[0].detail


def test_docs_automation_baseline_checker_accepts_required_files_and_markers(tmp_path: Path) -> None:
    checker_module = _load_checker_module()
    _seed_docs_automation_baseline(tmp_path, checker_module)

    violations = checker_module._find_docs_automation_baseline_violations(tmp_path)

    assert violations == []


def test_entry_point_guide_checker_requires_home_link_and_all_layers(tmp_path: Path) -> None:
    checker_module = _load_checker_module()
    _write_file(repo_root=tmp_path, relative_path="docs/index.rst", content="Home\n====\n")
    _write_file(
        repo_root=tmp_path,
        relative_path=checker_module.ENTRY_POINT_GUIDE_PATH,
        content="""
        ``DirectLLMCall``
        ``MultiStepAgent``
        patterns
        examples
        """,
    )

    violations = checker_module._find_entry_point_guide_violations(tmp_path)

    assert [violation.category for violation in violations] == [
        "entry-point-guide",
        "entry-point-guide",
    ]
    assert "docs/index.rst must link" in violations[0].detail
    assert "missing required marker '``Workflow``'" in violations[1].detail
