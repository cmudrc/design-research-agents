"""Tests for the Google-style docstring completeness checker."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_google_docstrings.py"


def _write_file(repo_root: Path, relative_path: str, content: str) -> None:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def _run_checker(repo_root: Path) -> tuple[int, list[str]]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--repo-root",
        str(repo_root),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return completed.returncode, lines


def _error_codes(lines: list[str]) -> set[str]:
    codes: set[str] = set()
    for line in lines:
        if "DGS" not in line:
            continue
        after_colon = line.split(": ", 1)[1]
        code = after_colon.split(" ", 1)[0]
        codes.add(code)
    return codes


def test_missing_module_class_and_callable_docstrings_fail() -> None:
    with_missing_docs = """
    class Example:
        def __init__(self, value: int) -> None:
            self.value = value

    def build(value: int) -> int:
        return value
    """
    repo_root = Path("artifacts/tests/docstring_checker_missing_docs")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/example.py", with_missing_docs)

    exit_code, lines = _run_checker(repo_root)

    assert exit_code == 1
    assert {"DGS001", "DGS003", "DGS005"} <= _error_codes(lines)


def test_missing_args_entries_are_reported() -> None:
    source = """
    \"\"\"Module summary.\"\"\"

    def transform(alpha: int, beta: int) -> int:
        \"\"\"Transform two values.

        Args:
            alpha: First value.

        Returns:
            The transformed result.
        \"\"\"
        return alpha + beta
    """
    repo_root = Path("artifacts/tests/docstring_checker_args")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/args_example.py", source)

    exit_code, lines = _run_checker(repo_root)

    assert exit_code == 1
    assert "DGS007" in _error_codes(lines)


def test_yields_and_returns_rules_apply_correctly() -> None:
    source = """
    \"\"\"Module summary.\"\"\"

    from collections.abc import Iterator

    def generate(limit: int) -> Iterator[int]:
        \"\"\"Yield values up to a limit.

        Args:
            limit: Inclusive upper bound.

        Returns:
            This section is intentionally incorrect for generators.
        \"\"\"
        for index in range(limit):
            yield index

    def compute(value: int) -> int:
        \"\"\"Compute an integer.

        Args:
            value: Input value.
        \"\"\"
        return value + 1
    """
    repo_root = Path("artifacts/tests/docstring_checker_yields_returns")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/returns_example.py", source)

    exit_code, lines = _run_checker(repo_root)
    codes = _error_codes(lines)

    assert exit_code == 1
    assert "DGS008" in codes
    assert "DGS009" in codes


def test_raises_rule_is_enforced() -> None:
    source = """
    \"\"\"Module summary.\"\"\"

    def explode(flag: bool) -> int:
        \"\"\"Raise when the input flag is true.

        Args:
            flag: Whether to raise.

        Returns:
            One when not raising.
        \"\"\"
        if flag:
            raise ValueError("boom")
        return 1
    """
    repo_root = Path("artifacts/tests/docstring_checker_raises")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/raises_example.py", source)

    exit_code, lines = _run_checker(repo_root)

    assert exit_code == 1
    assert "DGS010" in _error_codes(lines)


def test_varargs_kwargs_and_noreturn_are_supported() -> None:
    source = """
    \"\"\"Module summary.\"\"\"

    from typing import NoReturn

    def collect(*args: object, **kwargs: object) -> dict[str, object]:
        \"\"\"Collect dynamic arguments.

        Args:
            args: Positional values.
            kwargs: Keyword values.

        Returns:
            A copy of the keyword arguments.
        \"\"\"
        return dict(kwargs)

    def stop(message: str) -> NoReturn:
        \"\"\"Stop by raising an exception.

        Args:
            message: Error message.

        Raises:
            RuntimeError: Always raised.
        \"\"\"
        raise RuntimeError(message)
    """
    repo_root = Path("artifacts/tests/docstring_checker_varargs")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/varargs_example.py", source)

    exit_code, lines = _run_checker(repo_root)

    assert exit_code == 0
    assert lines == ["Google-style docstring checks passed."]


def test_tests_directory_is_out_of_scope() -> None:
    valid_source = """
    \"\"\"Module summary.\"\"\"

    def ready() -> int:
        \"\"\"Return a value.

        Returns:
            A constant integer.
        \"\"\"
        return 1
    """
    missing_doc_source = """
    def no_doc(value: int) -> int:
        return value
    """
    repo_root = Path("artifacts/tests/docstring_checker_scope")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/valid.py", valid_source)
    _write_file(repo_root, "tests/test_ignore_me.py", missing_doc_source)

    exit_code, lines = _run_checker(repo_root)

    assert exit_code == 0
    assert lines == ["Google-style docstring checks passed."]


def test_dataclass_fields_require_inline_docstrings() -> None:
    source = """
    \"\"\"Module summary.\"\"\"

    from dataclasses import dataclass
    from typing import ClassVar

    @dataclass
    class Example:
        \"\"\"Example dataclass.\"\"\"

        name: str
        count: int = 0
        \"\"\"Number of items.\"\"\"
        metadata: ClassVar[dict[str, int]] = {}
    """
    repo_root = Path("artifacts/tests/docstring_checker_dataclass_fields")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/dataclass_example.py", source)

    exit_code, lines = _run_checker(repo_root)

    assert exit_code == 1
    assert "DGS011" in _error_codes(lines)


def test_dataclass_alias_and_field_docstrings_pass() -> None:
    source = """
    \"\"\"Module summary.\"\"\"

    from dataclasses import dataclass as dc

    @dc
    class Example:
        \"\"\"Example dataclass.\"\"\"

        name: str
        \"\"\"Human-readable name.\"\"\"
        count: int = 0
        \"\"\"Number of items.\"\"\"
    """
    repo_root = Path("artifacts/tests/docstring_checker_dataclass_alias")
    if repo_root.exists():
        shutil.rmtree(repo_root)
    _write_file(repo_root, "src/pkg/dataclass_alias_example.py", source)

    exit_code, lines = _run_checker(repo_root)

    assert exit_code == 0
    assert lines == ["Google-style docstring checks passed."]
