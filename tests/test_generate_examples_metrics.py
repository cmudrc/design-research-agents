"""Tests for deterministic examples metrics generation."""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_examples_metrics.py"


def _load_metrics_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_examples_metrics", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_junit_xml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "examples.junit.xml"
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return path


def test_runnable_example_counts_ignore_non_runtime_example_checks(tmp_path: Path) -> None:
    metrics_module = _load_metrics_module()
    junit_xml = _write_junit_xml(
        tmp_path,
        """
        <testsuite tests="4">
          <testcase name="test_non_streaming_example_runs[examples/agents/alpha.py]" />
          <testcase name="test_execution_result_examples_build_summary_without_post_mutation[examples/agents/alpha.py]">
            <failure message="ignored extra assertion">failed</failure>
          </testcase>
          <testcase name="test_non_streaming_example_runs[examples/agents/beta.py]" />
          <testcase name="test_script_shell_example_runs" />
        </testsuite>
        """,
    )

    counts = metrics_module._parse_junit_runnable_example_counts(
        junit_xml,
        python_examples=(
            metrics_module.REPO_ROOT / "examples" / "agents" / "alpha.py",
            metrics_module.REPO_ROOT / "examples" / "agents" / "beta.py",
        ),
        shell_examples=(metrics_module.REPO_ROOT / "examples" / "tools" / "script_tools" / "tool.sh",),
    )

    assert counts == (3, 3, 0)


def test_runnable_example_counts_require_all_expected_python_examples(tmp_path: Path) -> None:
    metrics_module = _load_metrics_module()
    junit_xml = _write_junit_xml(
        tmp_path,
        """
        <testsuite tests="2">
          <testcase name="test_non_streaming_example_runs[examples/agents/alpha.py]" />
          <testcase name="test_script_shell_example_runs" />
        </testsuite>
        """,
    )

    with pytest.raises(ValueError, match="Deterministic Python example junit cases do not match"):
        metrics_module._parse_junit_runnable_example_counts(
            junit_xml,
            python_examples=(
                metrics_module.REPO_ROOT / "examples" / "agents" / "alpha.py",
                metrics_module.REPO_ROOT / "examples" / "agents" / "beta.py",
            ),
            shell_examples=(metrics_module.REPO_ROOT / "examples" / "tools" / "script_tools" / "tool.sh",),
        )


def test_public_api_symbols_are_read_from_export_manifest(tmp_path: Path) -> None:
    metrics_module = _load_metrics_module()
    manifest = tmp_path / "_public_exports.py"
    manifest.write_text(
        textwrap.dedent(
            """
            TOP_LEVEL_EXPORTS = {
                "Agent": "package:Agent",
                "Workflow": "package:Workflow",
            }

            TOP_LEVEL_SUBMODULES = {
                "study": "package.study",
            }
            """
        ).lstrip(),
        encoding="utf-8",
    )

    assert metrics_module._extract_public_api_symbols(manifest) == (
        "__version__",
        "Agent",
        "Workflow",
        "study",
    )
