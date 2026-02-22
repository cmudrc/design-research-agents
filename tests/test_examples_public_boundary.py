"""Boundary tests enforcing public-only example imports and client access patterns."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
REQUIRED_DOC_SECTIONS = (
    "Motivation",
    "Diagram",
    "Technical Walkthrough",
    "Expected Results",
    "Discussion",
)


def _collect_violations(*, pattern: re.Pattern[str], root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
    return violations


def _missing_doc_sections(lines: list[str]) -> list[str]:
    normalized = {line.strip() for line in lines if line.strip()}
    return [section for section in REQUIRED_DOC_SECTIONS if section not in normalized]


def _leading_shell_comment_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1

    collected: list[str] = []
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            collected.append(stripped[1:].strip())
            index += 1
            continue
        if not stripped:
            index += 1
            continue
        break
    return collected


def test_examples_do_not_import_internal_package_modules() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+design_research_agents\._")
    violations = _collect_violations(pattern=pattern, root=EXAMPLES_ROOT)
    assert violations == [], "\n".join(violations)


def test_client_examples_do_not_access_private_client_or_backend_attributes() -> None:
    client_examples_root = EXAMPLES_ROOT / "clients"
    pattern = re.compile(r"\bclient\._|\bbackend\._|\._vllm_server|\._ollama_server|\._sglang_server|\._llama_server")
    violations = _collect_violations(pattern=pattern, root=client_examples_root)
    assert violations == [], "\n".join(violations)


def test_examples_do_not_use_result_output_isinstance_guard_pattern() -> None:
    pattern = re.compile(r"result\.output if isinstance\(result\.output, dict\) else \{\}")
    violations = _collect_violations(pattern=pattern, root=EXAMPLES_ROOT)
    assert violations == [], "\n".join(violations)


def test_examples_do_not_use_hasattr_result_guard_pattern() -> None:
    pattern = re.compile(r"hasattr\(result,\s*[\"']success[\"']\)|unexpected result")
    violations = _collect_violations(pattern=pattern, root=EXAMPLES_ROOT)
    assert violations == [], "\n".join(violations)


def test_examples_do_not_include_support_helper_modules() -> None:
    helper_files = sorted(path.relative_to(REPO_ROOT).as_posix() for path in EXAMPLES_ROOT.rglob("_support_*.py"))
    assert helper_files == [], "\n".join(helper_files)


def test_examples_include_canonical_docs_sections() -> None:
    violations: list[str] = []

    for path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("_"):
            continue
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstring = ast.get_docstring(module, clean=False)
        if not isinstance(docstring, str) or not docstring.strip():
            violations.append(f"{path.relative_to(REPO_ROOT)}: missing module docstring")
            continue
        missing_sections = _missing_doc_sections(docstring.splitlines())
        if missing_sections:
            violations.append(f"{path.relative_to(REPO_ROOT)}: missing sections {missing_sections}")

    for path in sorted(EXAMPLES_ROOT.rglob("*.sh")):
        if "__pycache__" in path.parts or path.name.startswith("_"):
            continue
        comment_lines = _leading_shell_comment_lines(path)
        missing_sections = _missing_doc_sections(comment_lines)
        if missing_sections:
            violations.append(f"{path.relative_to(REPO_ROOT)}: missing sections {missing_sections}")

    assert violations == [], "\n".join(violations)


def test_generated_example_docs_are_up_to_date() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_example_docs.py", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Generated example docs are out of date.\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
