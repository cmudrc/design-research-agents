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
    """Return all runnable example scripts under ``examples/``."""
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
    """Return runnable Python example files."""
    return tuple(path for path in _discover_runnable_examples() if path.suffix == ".py")


def _parse_junit_pass_fail_counts(path: Path) -> tuple[int, int, int]:
    """Parse junit XML and return ``(total, passed, failed)`` testcase counts."""
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
    """Extract intended public symbols from ``design_research_agents.__init__``."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    export_dict: ast.Dict | None = None

    for node in module.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_EXPORTS":
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

    exports: list[str] = ["__version__"]
    for key in export_dict.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ValueError("_EXPORTS contains a non-string key, which is unsupported.")
        exports.append(key.value)
    return tuple(exports)


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
