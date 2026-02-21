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
BASELINE_ENTRY_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):\s+(?P<code>DGS\d+)\s+(?P<message>.+)$"
)
SUMMARY_PLACEHOLDER_PATTERN = re.compile(r"^run\s+[a-z_][a-z0-9_]*\.$")
PLACEHOLDER_DETAIL_TEXT = {
    "parameter value.",
    "the resulting value.",
    "raised when execution fails.",
}
PLACEHOLDER_VIOLATION_CODES = {"DGS014", "DGS015"}


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


def _is_placeholder_summary(summary: str) -> bool:
    """Return whether one summary line appears to be placeholder text."""
    normalized = summary.strip().lower()
    if not normalized:
        return False
    return bool(SUMMARY_PLACEHOLDER_PATTERN.match(normalized))


def _line_looks_like_placeholder_detail(line: str) -> bool:
    """Return whether one section line appears to be placeholder detail text."""
    stripped = line.strip()
    if not stripped:
        return False
    normalized_text = stripped.lower()
    if normalized_text in PLACEHOLDER_DETAIL_TEXT:
        return True
    args_match = ARGS_ENTRY_PATTERN.match(stripped)
    if args_match is None:
        return False
    _, _, after_colon = stripped.partition(":")
    return after_colon.strip().lower() in PLACEHOLDER_DETAIL_TEXT


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
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    target_types: tuple[type[ast.AST], ...],
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
            Violation(
                relative_path,
                1,
                "DGS002",
                "Module docstring must include a summary line.",
            )
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
    summary = _extract_summary(callable_docstring)
    if _is_placeholder_summary(summary):
        violations.append(
            Violation(
                relative_path,
                node.lineno,
                "DGS014",
                "Callable docstring summary appears to be placeholder text.",
            )
        )

    for section_name, section_lines in sections.items():
        if any(_line_looks_like_placeholder_detail(line) for line in section_lines):
            violations.append(
                Violation(
                    relative_path,
                    node.lineno,
                    "DGS015",
                    (
                        f"Callable docstring section '{section_name}' contains "
                        "placeholder detail text."
                    ),
                )
            )

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
                statement.body,
                relative_path,
                violations,
                decorator_names,
                module_aliases,
            )
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_validate_callable_docstring(statement, relative_path))
            _walk_node_body(
                statement.body,
                relative_path,
                violations,
                decorator_names,
                module_aliases,
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


def _load_baseline_entries(baseline_path: Path | None) -> set[str]:
    """Load baseline violation lines from one optional file.

    Args:
        baseline_path: Optional path to a newline-delimited violation baseline.

    Returns:
        Set of formatted violation lines to suppress.
    """
    if baseline_path is None:
        return set()
    if not baseline_path.exists():
        return set()
    entries: set[str] = set()
    for raw_line in baseline_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.add(line)
    return entries


def _normalize_repo_relative_path(path_text: str, repo_root: Path) -> str:
    """Normalize one path string into repository-relative POSIX style when possible.

    Args:
        path_text: Raw path text from CLI input files.
        repo_root: Repository root path for absolute-path normalization.

    Returns:
        Normalized path text.
    """
    path = Path(path_text)
    if path.is_absolute():
        try:
            resolved = path.resolve().relative_to(repo_root)
            return resolved.as_posix()
        except ValueError:
            return path.as_posix()
    normalized = path.as_posix()
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _load_changed_files(changed_files_path: Path | None, repo_root: Path) -> set[str]:
    """Load changed file paths from one optional newline-delimited file.

    Args:
        changed_files_path: Optional path containing changed repo file paths.
        repo_root: Repository root path used for path normalization.

    Returns:
        Set of normalized repository-relative paths.
    """
    if changed_files_path is None:
        return set()
    if not changed_files_path.exists():
        return set()

    changed_files: set[str] = set()
    for raw_line in changed_files_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        changed_files.add(_normalize_repo_relative_path(line, repo_root))
    return changed_files


def _baseline_entry_path(entry: str, repo_root: Path) -> str | None:
    """Extract normalized path from one formatted baseline entry line.

    Args:
        entry: Baseline violation line.
        repo_root: Repository root path used for path normalization.

    Returns:
        Parsed normalized path, or ``None`` for unparsable entries.
    """
    match = BASELINE_ENTRY_PATTERN.match(entry)
    if match is None:
        return None
    return _normalize_repo_relative_path(match.group("path"), repo_root)


def _collect_changed_file_baseline_violations(
    baseline_entries: set[str], changed_files: set[str], repo_root: Path
) -> tuple[list[Violation], set[str]]:
    """Collect guard violations for changed files that appear in baseline suppressions.

    Args:
        baseline_entries: Set of baseline suppression lines.
        changed_files: Set of changed repository-relative file paths.
        repo_root: Repository root path used for baseline path normalization.

    Returns:
        Tuple containing:
        - guard violations for changed files that rely on baseline suppressions
        - changed-file paths that should not receive baseline suppression
    """
    if not changed_files:
        return [], set()

    baselined_paths: set[str] = set()
    for entry in baseline_entries:
        entry_path = _baseline_entry_path(entry, repo_root)
        if entry_path is None:
            continue
        baselined_paths.add(entry_path)

    blocked_paths = sorted(path for path in changed_files if path in baselined_paths)
    violations = [
        Violation(
            path=path,
            line=1,
            code="DGS013",
            message=(
                "Changed file cannot rely on baseline suppressions; "
                "remove matching baseline entries."
            ),
        )
        for path in blocked_paths
    ]
    return violations, set(blocked_paths)


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
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional newline-delimited baseline file of known violations to suppress.",
    )
    parser.add_argument(
        "--changed-files-file",
        default=None,
        help=(
            "Optional newline-delimited file of changed repo-relative paths. "
            "Changed files are not allowed to rely on baseline suppressions."
        ),
    )
    parser.add_argument(
        "--enforce-codes",
        default=None,
        help=(
            "Optional comma-delimited violation-code allowlist. "
            "When provided, only matching codes are reported."
        ),
    )
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    baseline_path = Path(args.baseline).resolve() if args.baseline is not None else None
    changed_files_path = (
        Path(args.changed_files_file).resolve() if args.changed_files_file is not None else None
    )
    enforced_codes: set[str] | None = None
    if args.enforce_codes is not None:
        parsed_codes = [
            code.strip().upper() for code in str(args.enforce_codes).split(",") if code.strip()
        ]
        enforced_codes = set(parsed_codes)
    violations = _collect_violations(repo_root)
    baseline_entries = _load_baseline_entries(baseline_path)
    changed_files = _load_changed_files(changed_files_path, repo_root)
    changed_files_scope_enabled = changed_files_path is not None
    baseline_guard_violations, blocked_paths = _collect_changed_file_baseline_violations(
        baseline_entries=baseline_entries,
        changed_files=changed_files,
        repo_root=repo_root,
    )

    effective_baseline_entries: set[str] = set()
    for entry in baseline_entries:
        entry_path = _baseline_entry_path(entry, repo_root)
        if entry_path is not None and entry_path in blocked_paths:
            continue
        effective_baseline_entries.add(entry)

    unexpected = sorted(
        [
            *baseline_guard_violations,
            *[
                violation
                for violation in violations
                if not (
                    changed_files_scope_enabled
                    and violation.code in PLACEHOLDER_VIOLATION_CODES
                    and violation.path not in changed_files
                )
                if violation.format() not in effective_baseline_entries
            ],
        ],
        key=lambda item: (item.path, item.line, item.code, item.message),
    )
    if enforced_codes is not None:
        unexpected = [item for item in unexpected if item.code in enforced_codes]

    if not unexpected:
        print("Google-style docstring checks passed.")
        return 0

    for violation in unexpected:
        print(violation.format())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
