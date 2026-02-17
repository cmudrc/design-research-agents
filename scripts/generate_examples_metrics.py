#!/usr/bin/env python3
"""Generate deterministic examples metrics and public-API coverage metrics."""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
PUBLIC_API_INIT = REPO_ROOT / "src" / "design_research_agents" / "__init__.py"
JUNIT_XML = REPO_ROOT / "artifacts" / "examples" / "examples-deterministic.junit.xml"
METRICS_JSON = REPO_ROOT / "artifacts" / "examples" / "examples_metrics.json"


def _discover_runnable_examples() -> tuple[Path, ...]:
    """Run discover runnable examples.

    Returns:
        The resulting value.
    """
    discovered: list[Path] = []
    for extension in ("*.py", "*.sh"):
        for path in sorted(EXAMPLES_ROOT.rglob(extension)):
            parts = path.relative_to(REPO_ROOT).parts
            if "__pycache__" in parts:
                continue
            if path.name.startswith("_"):
                continue
            discovered.append(path)
    return tuple(sorted(discovered))


def _discover_python_examples() -> tuple[Path, ...]:
    """Run discover python examples.

    Returns:
        The resulting value.
    """
    return tuple(path for path in _discover_runnable_examples() if path.suffix == ".py")


def _parse_junit_pass_fail_counts(path: Path) -> tuple[int, int, int]:
    """Run parse junit pass fail counts.

    Args:
        path: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    testcases = list(root.iter("testcase"))
    if not testcases:
        raise ValueError(f"No testcase elements found in {path}.")

    passed = 0
    for testcase in testcases:
        non_passing = any(child.tag in {"failure", "error", "skipped"} for child in testcase)
        if not non_passing:
            passed += 1

    total = len(testcases)
    failed = total - passed
    return total, passed, failed


def _extract_exports_from_init(path: Path) -> tuple[str, ...]:
    """Run extract exports from init.

    Args:
        path: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    export_dict: ast.Dict | None = None

    for node in module.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_EXPORTS"
        ):
            if isinstance(node.value, ast.Dict):
                export_dict = node.value
            break
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "_EXPORTS":
                if isinstance(node.value, ast.Dict):
                    export_dict = node.value
                break

    if export_dict is None:
        raise ValueError(f"Unable to locate _EXPORTS dictionary in {path}.")

    exports: list[str] = []
    for key in export_dict.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ValueError("_EXPORTS contains a non-string key, which is unsupported.")
        exports.append(key.value)
    return tuple(exports)


def _collect_public_api_symbols_used_in_examples(
    example_files: tuple[Path, ...],
    export_symbols: tuple[str, ...],
) -> tuple[str, ...]:
    """Run collect public api symbols used in examples.

    Args:
        example_files: Parameter value.
        export_symbols: Parameter value.

    Returns:
        The resulting value.
    """
    export_set = set(export_symbols)
    covered: set[str] = set()

    for path in example_files:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package_aliases = _collect_package_aliases(module)
        covered.update(_collect_explicit_imported_exports(module, export_set))
        covered.update(_collect_attribute_access_exports(module, export_set, package_aliases))

    return tuple(sorted(covered))


def _collect_package_aliases(module: ast.Module) -> set[str]:
    """Run collect package aliases.

    Args:
        module: Parameter value.

    Returns:
        The resulting value.
    """
    aliases: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name == "design_research_agents":
                aliases.add(alias.asname or "design_research_agents")
    return aliases


def _collect_explicit_imported_exports(module: ast.Module, export_set: set[str]) -> set[str]:
    """Run collect explicit imported exports.

    Args:
        module: Parameter value.
        export_set: Parameter value.

    Returns:
        The resulting value.
    """
    covered: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_name = node.module or ""
        if module_name != "design_research_agents" and not module_name.startswith(
            "design_research_agents."
        ):
            continue
        for alias in node.names:
            imported_name = alias.name
            if imported_name == "*":
                covered.update(export_set)
                continue
            if imported_name in export_set:
                covered.add(imported_name)
    return covered


def _collect_attribute_access_exports(
    module: ast.Module,
    export_set: set[str],
    package_aliases: set[str],
) -> set[str]:
    """Run collect attribute access exports.

    Args:
        module: Parameter value.
        export_set: Parameter value.
        package_aliases: Parameter value.

    Returns:
        The resulting value.
    """
    covered: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id in package_aliases and node.attr in export_set:
            covered.add(node.attr)
    return covered


def _percent(part: int, whole: int) -> float:
    """Run percent.

    Args:
        part: Parameter value.
        whole: Parameter value.

    Returns:
        The resulting value.
    """
    if whole == 0:
        return 100.0
    return round((part / whole) * 100, 1)


def main() -> None:
    """Compute and persist examples pass/fail and API-in-examples coverage metrics.

    Raises:
        Exception: Raised when execution fails.
    """
    runnable_examples = _discover_runnable_examples()
    python_examples = _discover_python_examples()

    tests_total, tests_passed, tests_failed = _parse_junit_pass_fail_counts(JUNIT_XML)
    expected_total = len(runnable_examples)
    if tests_total != expected_total:
        raise ValueError(
            "Deterministic examples junit count does not match discovered runnable examples: "
            f"tests={tests_total}, discovered={expected_total}."
        )

    export_symbols = _extract_exports_from_init(PUBLIC_API_INIT)
    covered_symbols = _collect_public_api_symbols_used_in_examples(
        example_files=python_examples,
        export_symbols=export_symbols,
    )
    missing_symbols = tuple(sorted(set(export_symbols) - set(covered_symbols)))

    metrics = {
        "examples": {
            "total": tests_total,
            "passed": tests_passed,
            "failed": tests_failed,
            "pass_percent": _percent(tests_passed, tests_total),
        },
        "public_api": {
            "total_exports": len(export_symbols),
            "covered_exports": len(covered_symbols),
            "coverage_percent": _percent(len(covered_symbols), len(export_symbols)),
            "covered_symbols": list(covered_symbols),
            "missing_symbols": list(missing_symbols),
        },
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
