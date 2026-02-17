"""Enforce complete Google-style docstrings for source, examples, and scripts."""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCAN_ROOTS = ("src", "examples", "scripts")
SECTION_HEADER_PATTERN = re.compile(r"^([A-Za-z][A-Za-z ]+):\s*$")
ARGS_ENTRY_PATTERN = re.compile(r"^\s*(\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?:\s+.+$")


@dataclass(slots=True, frozen=True)
class Violation:
    """Represents a single docstring policy violation."""

    path: str
    """Repository-relative file path containing the violation."""
    line: int
    """One-based line number where the violation is reported."""
    code: str
    """Stable violation code used for automation and filtering."""
    message: str
    """Human-readable description of the policy violation."""

    def format(self) -> str:
        """Render a deterministic CLI-friendly violation line.

        Returns:
            Formatted violation text.
        """
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def _iter_python_files(repo_root: Path) -> list[Path]:
    """Collect scoped Python files for docstring enforcement.

    Args:
        repo_root: Repository root directory.

    Returns:
        Sorted Python files under configured scan roots.
    """
    python_files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for candidate in root.rglob("*.py"):
            if "__pycache__" in candidate.parts:
                continue
            python_files.append(candidate)
    return sorted(python_files)


def _extract_summary(docstring: str | None) -> str:
    """Extract the first prose summary line from a docstring.

    Args:
        docstring: Raw cleaned docstring text.

    Returns:
        First non-section summary line, or an empty string.
    """
    if not docstring:
        return ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_HEADER_PATTERN.match(stripped):
            continue
        return stripped
    return ""


def _parse_docstring_sections(docstring: str) -> dict[str, list[str]]:
    """Parse a Google-style docstring into section-line mappings.

    Args:
        docstring: Cleaned docstring text.

    Returns:
        Mapping of lowercase section names to section lines.
    """
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in docstring.splitlines():
        stripped = line.strip()
        section_match = SECTION_HEADER_PATTERN.match(stripped)
        if section_match:
            current_section = section_match.group(1).lower()
            sections.setdefault(current_section, [])
            continue
        if current_section is None:
            continue
        sections[current_section].append(line)
    return sections


def _args_section_names(section_lines: Iterable[str]) -> set[str]:
    """Extract parameter names documented in Args sections.

    Args:
        section_lines: Lines belonging to one or more Args sections.

    Returns:
        Normalized parameter names with and without vararg markers.
    """
    names: set[str] = set()
    for line in section_lines:
        match = ARGS_ENTRY_PATTERN.match(line)
        if not match:
            continue
        raw_name = match.group(1)
        names.add(raw_name)
        names.add(raw_name.lstrip("*"))
    return names


def _expected_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Build expected parameter list for callable Args sections.

    Args:
        node: Callable AST node.

    Returns:
        Parameter names in signature order.
    """
    params: list[str] = []
    all_named_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    for arg in all_named_args:
        if arg.arg in {"self", "cls"}:
            continue
        params.append(arg.arg)
    if node.args.vararg is not None:
        params.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg is not None:
        params.append(f"**{node.args.kwarg.arg}")
    return params


def _is_documented_parameter(parameter: str, documented_names: set[str]) -> bool:
    """Determine whether a parameter is documented in Args.

    Args:
        parameter: Expected parameter name from the signature.
        documented_names: Parsed Args entry names.

    Returns:
        True when parameter documentation exists.
    """
    if parameter in documented_names:
        return True
    if parameter.startswith("**"):
        return parameter[2:] in documented_names
    if parameter.startswith("*"):
        return parameter[1:] in documented_names
    return False


def _contains_node_type(
    node: ast.FunctionDef | ast.AsyncFunctionDef, target_types: tuple[type[ast.AST], ...]
) -> bool:
    """Check whether callable body contains one of the requested node types.

    Args:
        node: Callable AST node.
        target_types: Node types to search for.

    Returns:
        True when a matching node exists in the callable body.
    """
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if current is not node and isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        if isinstance(current, target_types):
            return True
        stack.extend(ast.iter_child_nodes(current))
    return False


def _is_generator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Determine whether callable yields values.

    Args:
        node: Callable AST node.

    Returns:
        True when callable contains ``yield`` or ``yield from``.
    """
    return _contains_node_type(node, (ast.Yield, ast.YieldFrom))


def _has_explicit_raise(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Determine whether callable explicitly raises exceptions.

    Args:
        node: Callable AST node.

    Returns:
        True when callable contains ``raise`` statements.
    """
    return _contains_node_type(node, (ast.Raise,))


def _normalized_annotation(annotation: ast.expr | None) -> str:
    """Normalize return annotation into comparable text.

    Args:
        annotation: Return annotation AST expression.

    Returns:
        Normalized annotation string.
    """
    if annotation is None:
        return ""
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return "None"
    text = ast.unparse(annotation).strip().strip("'\"")
    text = text.replace("typing.", "").replace("typing_extensions.", "")
    return text


def _requires_returns_section(node: ast.FunctionDef | ast.AsyncFunctionDef, yields: bool) -> bool:
    """Evaluate whether callable must include a Returns section.

    Args:
        node: Callable AST node.
        yields: Whether callable yields values.

    Returns:
        True when Returns section is required.
    """
    if yields:
        return False
    normalized = _normalized_annotation(node.returns)
    if not normalized:
        return False
    return normalized not in {"None", "NoReturn", "Never"}


def _validate_module_docstring(tree: ast.Module, relative_path: str) -> list[Violation]:
    """Validate module-level docstring and summary.

    Args:
        tree: Parsed module AST.
        relative_path: File path relative to repository root.

    Returns:
        Violations found at module scope.
    """
    violations: list[Violation] = []
    module_docstring = ast.get_docstring(tree, clean=True)
    if not module_docstring:
        violations.append(Violation(relative_path, 1, "DGS001", "Missing module docstring."))
        return violations
    if not _extract_summary(module_docstring):
        violations.append(
            Violation(relative_path, 1, "DGS002", "Module docstring must include a summary line.")
        )
    return violations


def _validate_class_docstring(node: ast.ClassDef, relative_path: str) -> list[Violation]:
    """Validate class docstring and summary.

    Args:
        node: Class AST node.
        relative_path: File path relative to repository root.

    Returns:
        Violations found for class.
    """
    violations: list[Violation] = []
    class_docstring = ast.get_docstring(node, clean=True)
    if not class_docstring:
        violations.append(
            Violation(relative_path, node.lineno, "DGS003", "Missing class docstring.")
        )
        return violations
    if not _extract_summary(class_docstring):
        violations.append(
            Violation(
                relative_path,
                node.lineno,
                "DGS004",
                "Class docstring must include a summary line.",
            )
        )
    return violations


def _resolve_dataclass_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Resolve dataclass decorator names and module aliases from imports.

    Args:
        tree: Parsed module AST.

    Returns:
        Tuple containing:
        - decorator names that refer to ``dataclass``
        - module aliases that refer to ``dataclasses``
    """
    decorator_names = {"dataclass"}
    module_aliases = {"dataclasses"}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "dataclasses":
            for alias in statement.names:
                if alias.name == "dataclass":
                    decorator_names.add(alias.asname or alias.name)
            continue
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "dataclasses":
                    module_aliases.add(alias.asname or alias.name)
    return decorator_names, module_aliases


def _is_dataclass_decorator(
    decorator: ast.expr, decorator_names: set[str], module_aliases: set[str]
) -> bool:
    """Check whether a decorator expression resolves to ``dataclass``.

    Args:
        decorator: Decorator AST expression.
        decorator_names: Valid decorator name aliases.
        module_aliases: Valid ``dataclasses`` module aliases.

    Returns:
        True when expression is a dataclass decorator.
    """
    if isinstance(decorator, ast.Call):
        return _is_dataclass_decorator(decorator.func, decorator_names, module_aliases)
    if isinstance(decorator, ast.Name):
        return decorator.id in decorator_names
    if isinstance(decorator, ast.Attribute):
        if decorator.attr != "dataclass":
            return False
        if isinstance(decorator.value, ast.Name):
            return decorator.value.id in module_aliases
        return True
    return False


def _is_dataclass_class(
    node: ast.ClassDef, decorator_names: set[str], module_aliases: set[str]
) -> bool:
    """Determine whether a class declaration is decorated as a dataclass.

    Args:
        node: Class AST node.
        decorator_names: Valid decorator name aliases.
        module_aliases: Valid ``dataclasses`` module aliases.

    Returns:
        True when class is a dataclass.
    """
    return any(
        _is_dataclass_decorator(decorator, decorator_names, module_aliases)
        for decorator in node.decorator_list
    )


def _annotation_excluded_from_dataclass_field_docs(annotation: ast.expr | None) -> bool:
    """Check whether an annotation should be skipped for dataclass field docs.

    Args:
        annotation: Field annotation expression.

    Returns:
        True when annotation is ``ClassVar``, ``InitVar``, or ``KW_ONLY``.
    """
    if annotation is None:
        return False
    raw = ast.unparse(annotation)
    normalized = raw.replace("typing.", "").replace("dataclasses.", "")
    return (
        normalized == "ClassVar"
        or normalized.startswith("ClassVar[")
        or normalized == "InitVar"
        or normalized.startswith("InitVar[")
        or normalized == "KW_ONLY"
    )


def _is_literal_string_expression(statement: ast.stmt) -> bool:
    """Check whether a statement is a literal string expression.

    Args:
        statement: Statement node.

    Returns:
        True when statement is an expression containing a string literal.
    """
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _validate_dataclass_field_docstrings(node: ast.ClassDef, relative_path: str) -> list[Violation]:
    """Validate dataclass field-level docstrings.

    Args:
        node: Dataclass AST node.
        relative_path: File path relative to repository root.

    Returns:
        Violations for missing or empty dataclass field docstrings.
    """
    violations: list[Violation] = []
    body = node.body
    for index, statement in enumerate(body):
        if not isinstance(statement, ast.AnnAssign):
            continue
        if not isinstance(statement.target, ast.Name):
            continue
        if _annotation_excluded_from_dataclass_field_docs(statement.annotation):
            continue

        field_name = statement.target.id
        next_statement = body[index + 1] if index + 1 < len(body) else None
        if next_statement is None or not _is_literal_string_expression(next_statement):
            violations.append(
                Violation(
                    relative_path,
                    statement.lineno,
                    "DGS011",
                    f"Dataclass field '{field_name}' must include an inline docstring.",
                )
            )
            continue

        assert isinstance(next_statement, ast.Expr)  # for type narrowing
        assert isinstance(next_statement.value, ast.Constant)
        field_docstring = str(next_statement.value.value).strip()
        if field_docstring:
            continue
        violations.append(
            Violation(
                relative_path,
                statement.lineno,
                "DGS012",
                f"Dataclass field '{field_name}' docstring must include text.",
            )
        )
    return violations


def _validate_callable_docstring(
    node: ast.FunctionDef | ast.AsyncFunctionDef, relative_path: str
) -> list[Violation]:
    """Validate callable docstring completeness.

    Args:
        node: Callable AST node.
        relative_path: File path relative to repository root.

    Returns:
        Violations found for callable.
    """
    violations: list[Violation] = []
    callable_docstring = ast.get_docstring(node, clean=True)
    if not callable_docstring:
        violations.append(
            Violation(relative_path, node.lineno, "DGS005", "Missing callable docstring.")
        )
        return violations

    sections = _parse_docstring_sections(callable_docstring)
    expected_params = _expected_parameters(node)
    if expected_params:
        if "args" not in sections:
            violations.append(
                Violation(
                    relative_path,
                    node.lineno,
                    "DGS006",
                    "Callable with parameters must include an Args section.",
                )
            )
        else:
            documented = _args_section_names(sections["args"])
            for parameter in expected_params:
                if _is_documented_parameter(parameter, documented):
                    continue
                violations.append(
                    Violation(
                        relative_path,
                        node.lineno,
                        "DGS007",
                        f"Parameter '{parameter}' is missing from Args section.",
                    )
                )

    yields = _is_generator(node)
    if yields and "yields" not in sections:
        violations.append(
            Violation(
                relative_path,
                node.lineno,
                "DGS008",
                "Generator callable must include a Yields section.",
            )
        )

    if _requires_returns_section(node, yields) and "returns" not in sections:
        violations.append(
            Violation(
                relative_path,
                node.lineno,
                "DGS009",
                "Callable must include a Returns section for non-None return annotations.",
            )
        )

    if _has_explicit_raise(node) and "raises" not in sections:
        violations.append(
            Violation(
                relative_path,
                node.lineno,
                "DGS010",
                "Callable with explicit raise statements must include a Raises section.",
            )
        )
    return violations


def _walk_node_body(
    body: list[ast.stmt],
    relative_path: str,
    violations: list[Violation],
    decorator_names: set[str],
    module_aliases: set[str],
) -> None:
    """Walk statements recursively and validate classes/callables.

    Args:
        body: Statement list to inspect.
        relative_path: File path relative to repository root.
        violations: Mutable sink for collected violations.
        decorator_names: Resolved dataclass decorator aliases.
        module_aliases: Resolved dataclasses module aliases.

    Returns:
        None.
    """
    for statement in body:
        if isinstance(statement, ast.ClassDef):
            violations.extend(_validate_class_docstring(statement, relative_path))
            if _is_dataclass_class(statement, decorator_names, module_aliases):
                violations.extend(_validate_dataclass_field_docstrings(statement, relative_path))
            _walk_node_body(
                statement.body, relative_path, violations, decorator_names, module_aliases
            )
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_validate_callable_docstring(statement, relative_path))
            _walk_node_body(
                statement.body, relative_path, violations, decorator_names, module_aliases
            )


def _scan_file(file_path: Path, repo_root: Path) -> list[Violation]:
    """Parse and validate one Python file.

    Args:
        file_path: Absolute Python file path.
        repo_root: Repository root path.

    Returns:
        Violations detected in file.
    """
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=file_path.as_posix())
    decorator_names, module_aliases = _resolve_dataclass_aliases(tree)
    relative_path = file_path.relative_to(repo_root).as_posix()
    violations = _validate_module_docstring(tree, relative_path)
    _walk_node_body(tree.body, relative_path, violations, decorator_names, module_aliases)
    return violations


def _collect_violations(repo_root: Path) -> list[Violation]:
    """Collect docstring violations for all scoped files.

    Args:
        repo_root: Repository root path.

    Returns:
        Sorted violation list.
    """
    violations: list[Violation] = []
    for file_path in _iter_python_files(repo_root):
        violations.extend(_scan_file(file_path, repo_root))
    return sorted(violations, key=lambda item: (item.path, item.line, item.code, item.message))


def main() -> int:
    """Run docstring checks and return process status.

    Returns:
        ``0`` when checks pass, otherwise ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current working directory).",
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    violations = _collect_violations(repo_root)
    if not violations:
        print("Google-style docstring checks passed.")
        return 0

    for violation in violations:
        print(violation.format())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
