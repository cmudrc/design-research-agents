"""Boundary tests enforcing public-only example imports and client access patterns."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
FULL_REQUIRED_DOC_SECTIONS = (
    "Introduction",
    "Technical Implementation",
    "Expected Results",
    "References",
)
MINIMAL_REQUIRED_DOC_SECTIONS = (
    "Introduction",
    "Technical Implementation",
    "Expected Results",
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


def _iter_example_doc_lines() -> list[tuple[Path, list[str]]]:
    docs: list[tuple[Path, list[str]]] = []
    for path in sorted(EXAMPLES_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.name.startswith("_"):
            continue
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstring = ast.get_docstring(module, clean=False)
        docs.append((path, docstring.splitlines() if isinstance(docstring, str) and docstring.strip() else []))
    for path in sorted(EXAMPLES_ROOT.rglob("*.sh")):
        if "__pycache__" in path.parts or path.name.startswith("_"):
            continue
        docs.append((path, _leading_shell_comment_lines(path)))
    return docs


def _extract_markdown_section_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("## "):
            continue
        heading = line[3:].strip()
        if heading:
            names.add(heading)
    return names


def _parse_markdown_sections(lines: list[str]) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            current = heading if heading else None
            if current is not None:
                sections[current] = []
            continue
        if current is not None:
            sections[current].append(raw_line.rstrip())
    return {heading: "\n".join(body).strip() for heading, body in sections.items()}


def _missing_doc_sections(lines: list[str], *, required_sections: tuple[str, ...]) -> list[str]:
    present_sections = _extract_markdown_section_names(lines)
    return [section for section in required_sections if section not in present_sections]


def _is_script_tool_example(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT).as_posix()
    return relative.startswith("examples/tools/script_tools/")


def _normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


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

    for path, lines in _iter_example_doc_lines():
        required_sections = (
            MINIMAL_REQUIRED_DOC_SECTIONS if _is_script_tool_example(path) else FULL_REQUIRED_DOC_SECTIONS
        )
        missing_sections = _missing_doc_sections(lines, required_sections=required_sections)
        if missing_sections:
            violations.append(f"{path.relative_to(REPO_ROOT)}: missing sections {missing_sections}")

    assert violations == [], "\n".join(violations)


def test_non_script_examples_references_are_three_rst_links() -> None:
    link_pattern = re.compile(r"^- `[^`]+ <https?://[^>]+>`_$")
    violations: list[str] = []
    for path, lines in _iter_example_doc_lines():
        if _is_script_tool_example(path):
            continue
        sections = _parse_markdown_sections(lines)
        references = sections.get("References", "")
        entries = [line.strip() for line in references.splitlines() if line.strip()]
        if len(entries) != 3:
            violations.append(f"{path.relative_to(REPO_ROOT)}: expected 3 references, found {len(entries)}")
            continue
        for entry in entries:
            if link_pattern.fullmatch(entry) is None:
                violations.append(f"{path.relative_to(REPO_ROOT)}: invalid reference format: {entry}")
    assert violations == [], "\n".join(violations)


def test_non_script_examples_have_unique_introductions_and_references() -> None:
    introduction_seen: dict[str, Path] = {}
    references_seen: dict[str, Path] = {}
    violations: list[str] = []
    for path, lines in _iter_example_doc_lines():
        if _is_script_tool_example(path):
            continue
        sections = _parse_markdown_sections(lines)
        introduction_norm = _normalize_block(sections.get("Introduction", ""))
        references_norm = _normalize_block(sections.get("References", ""))
        previous_intro = introduction_seen.get(introduction_norm)
        if previous_intro is not None:
            violations.append(
                f"{path.relative_to(REPO_ROOT)}: Introduction duplicates {previous_intro.relative_to(REPO_ROOT)}"
            )
        else:
            introduction_seen[introduction_norm] = path
        previous_refs = references_seen.get(references_norm)
        if previous_refs is not None:
            violations.append(
                f"{path.relative_to(REPO_ROOT)}: References duplicate {previous_refs.relative_to(REPO_ROOT)}"
            )
        else:
            references_seen[references_norm] = path
    assert violations == [], "\n".join(violations)


def test_examples_do_not_include_legacy_expected_results_run_preface() -> None:
    violations: list[str] = []
    for path, lines in _iter_example_doc_lines():
        sections = _parse_markdown_sections(lines)
        expected = sections.get("Expected Results", "")
        body_lines = expected.splitlines()
        index = 0
        while index < len(body_lines) and not body_lines[index].strip():
            index += 1
        if index < len(body_lines) and body_lines[index].strip().lower() == "run:":
            violations.append(f"{path.relative_to(REPO_ROOT)}: Expected Results starts with legacy 'Run:' preface")
    assert violations == [], "\n".join(violations)


def test_examples_do_not_include_chatgpt_citation_artifacts() -> None:
    citation_pattern = re.compile(r"【\d+[^】]*†L\d+(?:-L\d+)?】")
    violations: list[str] = []
    for path in sorted(EXAMPLES_ROOT.rglob("*")):
        if path.suffix not in {".py", ".sh"}:
            continue
        if "__pycache__" in path.parts or path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if citation_pattern.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
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
