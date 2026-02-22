"""Boundary tests enforcing public-only example imports and client access patterns."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"


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


def test_examples_do_not_import_internal_package_modules() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+design_research_agents\._")
    violations = _collect_violations(pattern=pattern, root=EXAMPLES_ROOT)
    assert violations == [], "\n".join(violations)


def test_client_examples_do_not_access_private_client_or_backend_attributes() -> None:
    client_examples_root = EXAMPLES_ROOT / "clients"
    pattern = re.compile(r"\bclient\._|\bbackend\._|\._vllm_server|\._ollama_server|\._sglang_server|\._llama_server")
    violations = _collect_violations(pattern=pattern, root=client_examples_root)
    assert violations == [], "\n".join(violations)
