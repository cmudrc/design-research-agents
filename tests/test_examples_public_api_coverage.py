from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
PACKAGE_INIT = REPO_ROOT / "src" / "design_research_agents" / "__init__.py"


def _public_symbols() -> tuple[str, ...]:
    module = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"), filename=str(PACKAGE_INIT))
    exports: list[str] = []

    export_dict: ast.Dict | None = None
    for node in module.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_EXPORTS"
            and isinstance(node.value, ast.Dict)
        ):
            export_dict = node.value
            break
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "_EXPORTS" and isinstance(node.value, ast.Dict):
                export_dict = node.value
                break

    if export_dict is None:
        raise AssertionError("Unable to locate _EXPORTS in package __init__.py")

    for key in export_dict.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise AssertionError("_EXPORTS contains a non-string key")
        exports.append(key.value)
    return tuple(name for name in exports if not (name.startswith("__") and name.endswith("__")))


def _python_examples() -> tuple[Path, ...]:
    discovered: list[Path] = []
    for path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name.startswith("_"):
            continue
        discovered.append(path)
    return tuple(discovered)


def _collect_package_aliases(module: ast.Module) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name == "design_research_agents":
                aliases.add(alias.asname or "design_research_agents")
    return aliases


def _explicit_imported_symbols(module: ast.Module, exports: set[str]) -> set[str]:
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
                covered.update(exports)
                continue
            if imported_name in exports:
                covered.add(imported_name)
    return covered


def _attribute_symbols(module: ast.Module, exports: set[str], aliases: set[str]) -> set[str]:
    covered: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id in aliases and node.attr in exports:
            covered.add(node.attr)
    return covered


def test_examples_cover_all_curated_public_symbols() -> None:
    symbols = _public_symbols()
    export_set = set(symbols)
    covered: set[str] = set()
    symbol_to_examples: dict[str, list[str]] = {symbol: [] for symbol in symbols}

    for path in _python_examples():
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _collect_package_aliases(module)
        used = _explicit_imported_symbols(module, export_set) | _attribute_symbols(
            module,
            export_set,
            aliases,
        )
        relative = path.relative_to(REPO_ROOT).as_posix()
        for symbol in sorted(used):
            covered.add(symbol)
            symbol_to_examples[symbol].append(relative)

    missing = sorted(export_set - covered)
    assert not missing, (
        "Python examples must cover every curated public symbol. Missing: "
        + ", ".join(missing)
        + f".\nSymbol usage map: {symbol_to_examples}"
    )
