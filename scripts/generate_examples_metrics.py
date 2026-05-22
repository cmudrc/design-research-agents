#!/usr/bin/env python3
"""Generate deterministic examples metrics and public-API coverage metrics."""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
PUBLIC_API_EXPORTS = REPO_ROOT / "src" / "design_research_agents" / "_public_exports.py"
JUNIT_XML = REPO_ROOT / "artifacts" / "examples" / "examples-deterministic.junit.xml"
METRICS_JSON = REPO_ROOT / "artifacts" / "examples" / "examples_metrics.json"
_NON_STREAMING_TEST_PREFIX = "test_non_streaming_example_runs["
_SCRIPT_SHELL_TEST_NAME = "test_script_shell_example_runs"


def _discover_example_paths(*extensions: str, exclude_streaming: bool = False) -> tuple[Path, ...]:
    """Return example files under ``examples/`` matching ``extensions``."""
    discovered: list[Path] = []
    for extension in extensions:
        for path in sorted(EXAMPLES_ROOT.rglob(extension)):
            parts = path.relative_to(REPO_ROOT).parts
            if "__pycache__" in parts:
                continue
            if exclude_streaming and "streaming" in parts:
                continue
            if path.name.startswith("_"):
                continue
            discovered.append(path)
    return tuple(sorted(discovered))


def _discover_python_examples() -> tuple[Path, ...]:
    """Return runnable Python example files."""
    return _discover_example_paths("*.py")


def _discover_deterministic_python_examples() -> tuple[Path, ...]:
    """Return deterministic Python examples covered by ``tests/test_examples_non_streaming.py``."""
    return _discover_example_paths("*.py", exclude_streaming=True)


def _discover_shell_examples() -> tuple[Path, ...]:
    """Return runnable shell examples."""
    return _discover_example_paths("*.sh")


def _extract_pytest_parameterized_value(name: str, prefix: str) -> str | None:
    """Return the parameterized pytest value embedded in ``name`` when present."""
    if not name.startswith(prefix) or not name.endswith("]"):
        return None
    return name[len(prefix) : -1]


def _parse_junit_runnable_example_counts(
    path: Path,
    *,
    python_examples: tuple[Path, ...],
    shell_examples: tuple[Path, ...],
) -> tuple[int, int, int]:
    """Return pass/fail counts for example execution testcases in the junit report."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    testcases = tuple(root.iter("testcase"))
    if not testcases:
        raise ValueError(f"No testcase elements found in {path}.")

    expected_python = tuple(sorted(example.relative_to(REPO_ROOT).as_posix() for example in python_examples))
    observed_python: list[str] = []
    observed_shell = 0
    passed = 0

    for testcase in testcases:
        name = testcase.get("name", "")
        python_example = _extract_pytest_parameterized_value(name, _NON_STREAMING_TEST_PREFIX)
        if python_example is not None:
            observed_python.append(python_example)
            if not any(child.tag in {"failure", "error", "skipped"} for child in testcase):
                passed += 1
            continue
        if name == _SCRIPT_SHELL_TEST_NAME:
            observed_shell += 1
            if not any(child.tag in {"failure", "error", "skipped"} for child in testcase):
                passed += 1

    observed_python_sorted = tuple(sorted(observed_python))
    if observed_python_sorted != expected_python:
        raise ValueError(
            "Deterministic Python example junit cases do not match discovered runnable examples: "
            f"tests={observed_python_sorted}, discovered={expected_python}."
        )

    expected_shell = len(shell_examples)
    if observed_shell != expected_shell:
        raise ValueError(
            "Deterministic shell example junit count does not match discovered runnable examples: "
            f"tests={observed_shell}, discovered={expected_shell}."
        )

    total = len(expected_python) + expected_shell
    failed = total - passed
    return total, passed, failed


def _extract_string_keyed_dict(module: ast.Module, name: str, path: Path) -> tuple[str, ...]:
    """Extract ordered string keys from a module-level dictionary assignment."""
    export_dict: ast.Dict | None = None

    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if isinstance(node.value, ast.Dict):
                export_dict = node.value
            break
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(node.value, ast.Dict):
                    export_dict = node.value
                break

    if export_dict is None:
        raise ValueError(f"Unable to locate {name} dictionary in {path}.")

    keys: list[str] = []
    for key in export_dict.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ValueError(f"{name} contains a non-string key, which is unsupported.")
        keys.append(key.value)
    return tuple(keys)


def _extract_public_api_symbols(path: Path) -> tuple[str, ...]:
    """Extract intended top-level public symbols from the public export manifest."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    exports = _extract_string_keyed_dict(module, "TOP_LEVEL_EXPORTS", path)
    submodules = _extract_string_keyed_dict(module, "TOP_LEVEL_SUBMODULES", path)
    return ("__version__", *exports, *submodules)


def _collect_public_api_symbol_usage(
    example_files: tuple[Path, ...],
    export_symbols: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Map each public symbol to sorted example files that reference it."""
    export_set = set(export_symbols)
    usage_map: dict[str, set[str]] = {symbol: set() for symbol in export_symbols}

    for path in example_files:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package_aliases = _collect_package_aliases(module)
        used_symbols = set()
        used_symbols.update(_collect_explicit_imported_exports(module, export_set))
        used_symbols.update(_collect_attribute_access_exports(module, export_set, package_aliases))
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for symbol in used_symbols:
            usage_map[symbol].add(relative_path)

    return {symbol: tuple(sorted(paths)) for symbol, paths in sorted(usage_map.items())}


def _collect_package_aliases(module: ast.Module) -> set[str]:
    """Collect local aliases for direct ``import design_research_agents`` statements."""
    aliases: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name == "design_research_agents":
                aliases.add(alias.asname or "design_research_agents")
    return aliases


def _collect_explicit_imported_exports(module: ast.Module, export_set: set[str]) -> set[str]:
    """Collect exported symbols imported via ``from design_research_agents...``."""
    covered: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.ImportFrom):
            continue
        module_name = node.module or ""
        if module_name != "design_research_agents" and not module_name.startswith("design_research_agents."):
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
    """Collect exported symbols accessed as attributes from package aliases."""
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
    """Return percentage ``part/whole`` rounded to one decimal place."""
    if whole == 0:
        return 100.0
    return round((part / whole) * 100, 1)


def main() -> None:
    """Compute and write example-test and public-API coverage metrics."""
    deterministic_python_examples = _discover_deterministic_python_examples()
    shell_examples = _discover_shell_examples()
    python_examples = _discover_python_examples()

    tests_total, tests_passed, tests_failed = _parse_junit_runnable_example_counts(
        JUNIT_XML,
        python_examples=deterministic_python_examples,
        shell_examples=shell_examples,
    )

    export_symbols = _extract_public_api_symbols(PUBLIC_API_EXPORTS)
    symbol_usage = _collect_public_api_symbol_usage(
        example_files=python_examples,
        export_symbols=export_symbols,
    )
    covered_symbols = tuple(sorted(symbol for symbol, paths in symbol_usage.items() if paths))
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
            "symbol_to_examples": {symbol: list(paths) for symbol, paths in symbol_usage.items()},
        },
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {METRICS_JSON}")


if __name__ == "__main__":
    main()
