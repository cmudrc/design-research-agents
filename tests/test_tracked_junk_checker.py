"""Tests for tracked junk artifact detection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_no_tracked_junk.py"


def _load_checker_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_no_tracked_junk", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_offending_paths_flags_generated_artifacts() -> None:
    checker = _load_checker_module()

    offending = checker._find_offending_paths(
        [
            "docs/_build/html/index.html",
            "htmlcov/index.html",
            "src/design_research_agents/module.pyc",
            "src/design_research_agents/__pycache__/module.cpython-312.pyc",
            "src/design_research_agents/module.py",
        ]
    )

    assert offending == [
        "docs/_build/html/index.html",
        "htmlcov/index.html",
        "src/design_research_agents/module.pyc",
        "src/design_research_agents/__pycache__/module.cpython-312.pyc",
    ]


def test_find_offending_paths_ignores_normal_source_files() -> None:
    checker = _load_checker_module()

    offending = checker._find_offending_paths(
        [
            "docs/reference/index.rst",
            "src/design_research_agents/__init__.py",
            "tests/test_public_api.py",
        ]
    )

    assert offending == []
