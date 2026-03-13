#!/usr/bin/env python3
"""Generate per-example Sphinx pages from canonical example doc sections."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

FULL_REQUIRED_SECTIONS = (
    "Introduction",
    "Technical Implementation",
    "Expected Results",
    "References",
)

MINIMAL_REQUIRED_SECTIONS = (
    "Introduction",
    "Technical Implementation",
    "Expected Results",
)

ALL_SUPPORTED_SECTIONS = (
    "Introduction",
    "Technical Implementation",
    "Expected Results",
    "References",
)

CATEGORY_ORDER = (
    "agents",
    "workflow",
    "patterns",
    "clients",
    "model_selection",
    "tools",
    "optimization",
)

CATEGORY_TITLES = {
    "agents": "Agent Examples",
    "workflow": "Workflow Primitive Examples",
    "patterns": "Pattern Examples",
    "clients": "Client Examples",
    "model_selection": "Model Selection Examples",
    "tools": "Tool Examples",
    "optimization": "Optimization Examples",
}

TITLE_TOKEN_OVERRIDES = {
    "api": "API",
    "cpp": "CPP",
    "http": "HTTP",
    "json": "JSON",
    "llm": "LLM",
    "mcp": "MCP",
    "mlx": "MLX",
    "openai": "OpenAI",
    "sglang": "SGLang",
    "vllm": "vLLM",
}


@dataclass(slots=True, frozen=True)
class ExampleDocSpec:
    """One runnable example and parsed canonical docs content."""

    rel_path: str
    category: str
    slug: str
    title: str
    extension: str
    source_start_line: int
    sections: dict[str, str]


def _repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parents[1]


def _discover_runnable_examples(repo_root: Path) -> list[Path]:
    """Discover runnable example scripts under ``examples/``."""
    examples_root = repo_root / "examples"
    discovered: list[Path] = []
    for extension in ("*.py", "*.sh"):
        for path in sorted(examples_root.rglob(extension)):
            rel_parts = path.relative_to(examples_root).parts
            if "__pycache__" in rel_parts:
                continue
            if path.name.startswith("_"):
                continue
            discovered.append(path)
    return discovered


def _parse_python_doc_text(path: Path) -> tuple[str, int]:
    """Parse module docstring text and source start line from one Python example."""
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    docstring = ast.get_docstring(module, clean=False)
    if not isinstance(docstring, str) or not docstring.strip():
        raise ValueError(f"{path}: missing module docstring.")

    source_start_line = 1
    if module.body:
        first = module.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and isinstance(first.end_lineno, int)
        ):
            source_start_line = first.end_lineno + 1

    lines = source.splitlines()
    while source_start_line <= len(lines) and not lines[source_start_line - 1].strip():
        source_start_line += 1

    return docstring, source_start_line


def _parse_shell_doc_text(path: Path) -> str:
    """Parse leading comment block text from one shell example."""
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1

    collected: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("#"):
            collected.append(stripped[1:].strip())
            index += 1
            continue
        if not stripped:
            index += 1
            continue
        break

    doc_text = "\n".join(collected).strip()
    if not doc_text:
        raise ValueError(f"{path}: missing leading comment block.")
    return doc_text


def _is_script_tool_example(rel_path: str) -> bool:
    """Return True when one example path belongs to script tools."""
    return rel_path.startswith("examples/tools/script_tools/")


def _required_sections_for_example(rel_path: str) -> tuple[str, ...]:
    """Return required section profile for one example path."""
    if _is_script_tool_example(rel_path):
        return MINIMAL_REQUIRED_SECTIONS
    return FULL_REQUIRED_SECTIONS


def _parse_canonical_sections(
    *,
    doc_text: str,
    source_path: Path,
    required_sections: tuple[str, ...],
) -> dict[str, str]:
    """Parse canonical docs sections from one source doc block."""
    heading_pattern = re.compile(r"^##\s+(.+?)\s*$")
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in doc_text.splitlines():
        line = raw_line.rstrip()
        match = heading_pattern.match(line.strip())
        if match is not None:
            heading = match.group(1).strip()
            if heading in ALL_SUPPORTED_SECTIONS:
                current_section = heading
                sections[current_section] = []
            else:
                current_section = None
            continue
        if current_section is not None:
            sections[current_section].append(line)

    missing = [section for section in required_sections if section not in sections]
    if missing:
        raise ValueError(f"{source_path}: missing canonical section(s): {missing}")

    parsed = {name: "\n".join(sections[name]).strip() for name in sections}
    if "Technical Implementation" in required_sections:
        _extract_mermaid(parsed.get("Technical Implementation", ""), source_path=source_path.as_posix())
    return parsed


def _slug_for_example(*, rel_parts: tuple[str, ...], extension: str) -> str:
    """Build deterministic docs slug for one example path."""
    subpath_without_ext = Path(*rel_parts[1:]).with_suffix("").as_posix()
    slug = subpath_without_ext.replace("/", "_").replace("-", "_")
    if not slug:
        slug = Path(rel_parts[-1]).stem
    return slug


def _title_for_example(*, rel_parts: tuple[str, ...], extension: str) -> str:
    """Build human-readable page title for one example path."""
    del extension
    subpath_without_ext = Path(*rel_parts[1:]).with_suffix("").as_posix()
    label = subpath_without_ext.replace("/", " / ").replace("_", " ").replace("-", " ")
    title_parts: list[str] = []
    for token in label.split(" "):
        if token == "/":
            title_parts.append(token)
            continue
        normalized = token.strip().lower()
        if not normalized:
            continue
        title_parts.append(TITLE_TOKEN_OVERRIDES.get(normalized, normalized.capitalize()))
    return " ".join(title_parts)


def _extract_mermaid(technical_section: str, *, source_path: str) -> str:
    """Extract Mermaid diagram text from canonical Technical Implementation section."""
    lines = technical_section.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "```mermaid":
            start = index + 1
            break

    if start is not None:
        end = len(lines)
        for index in range(start, len(lines)):
            if lines[index].strip() == "```":
                end = index
                break
        mermaid_text = "\n".join(lines[start:end]).strip()
    else:
        mermaid_text = technical_section.strip()

    if not mermaid_text:
        raise ValueError(f"{source_path}: Technical Implementation must include Mermaid content.")
    return mermaid_text


def _strip_mermaid_block(technical_section: str) -> str:
    """Return Technical Implementation body without the Mermaid fenced block."""
    lines = technical_section.splitlines()
    start = None
    end = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "```mermaid":
            start = index
            break
    if start is None:
        return technical_section.strip()
    for index in range(start + 1, len(lines)):
        if lines[index].strip() == "```":
            end = index
            break
    if end is None:
        end = len(lines) - 1

    stripped_lines = lines[:start] + lines[end + 1 :]
    return "\n".join(stripped_lines).strip()


def _strip_expected_results_run_preface(expected_results: str) -> str:
    """Remove legacy run-command preface from Expected Results text."""
    lines = expected_results.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or lines[index].strip().lower() != "run:":
        return expected_results.strip()

    index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:]).strip()


def _normalize_code_block_indentation(body: str) -> str:
    """Ensure code-block bodies remain indented when embedded text contains newlines."""
    lines = body.splitlines()
    normalized: list[str] = []
    in_code_block = False
    saw_content_line = False

    for line in lines:
        stripped = line.strip()
        if not in_code_block and stripped.startswith(".. code-block::"):
            in_code_block = True
            saw_content_line = False
            normalized.append(line)
            continue

        if in_code_block:
            if not saw_content_line:
                normalized.append(line)
                if stripped:
                    saw_content_line = True
                continue
            if line.startswith("   ") or not stripped:
                normalized.append(line)
            else:
                normalized.append(f"   {line}")
            continue

        normalized.append(line)

    return "\n".join(normalized).strip()


def _render_optional_section(*, heading: str, body: str | None, prelude: list[str] | None = None) -> list[str]:
    """Render one optional RST section block."""
    normalized = (body or "").strip()
    if not normalized and not prelude:
        return []
    lines = [
        heading,
        "-" * len(heading),
        "",
    ]
    if prelude:
        lines.extend(prelude)
    if normalized:
        lines.extend(
            [
                normalized,
                "",
            ]
        )
    elif lines[-1] != "":
        lines.append("")
    return lines


def _render_example_page(spec: ExampleDocSpec) -> str:
    """Render one example page as RST."""
    title = spec.title
    run_command = f"PYTHONPATH=src python3 {spec.rel_path}" if spec.extension == ".py" else f"bash {spec.rel_path}"
    include_path = f"../../../{spec.rel_path}"
    literal_language = "python" if spec.extension == ".py" else "bash"

    introduction = spec.sections["Introduction"]
    technical_implementation = spec.sections["Technical Implementation"]
    expected_results = _normalize_code_block_indentation(
        _strip_expected_results_run_preface(spec.sections["Expected Results"])
    )
    references = spec.sections.get("References")
    mermaid = _extract_mermaid(technical_implementation, source_path=spec.rel_path)
    technical_text = _strip_mermaid_block(technical_implementation)
    indented_mermaid = "\n".join(f"   {line}" for line in mermaid.splitlines())

    source_block = [
        f".. literalinclude:: {include_path}",
        f"   :language: {literal_language}",
    ]
    if spec.extension == ".py":
        source_block.append(f"   :lines: {spec.source_start_line}-")
    source_block.extend(
        [
            "   :linenos:",
            "",
        ]
    )

    diagram_block = [
        ".. mermaid::",
        "",
        indented_mermaid,
        "",
    ]

    lines = [
        title,
        "=" * len(title),
        "",
        f"Source: ``{spec.rel_path}``",
        "",
        "Introduction",
        "------------",
        "",
        introduction,
        "",
    ]
    lines.extend(
        [
            "Technical Implementation",
            "------------------------",
            "",
        ]
    )
    if technical_text:
        lines.extend(
            [
                technical_text,
                "",
            ]
        )
    lines.extend(diagram_block)
    lines.extend(source_block)
    expected_prelude = [
        ".. rubric:: Run Command",
        "",
        ".. code-block:: bash",
        "",
        f"   {run_command}",
        "",
    ]
    lines.extend(_render_optional_section(heading="Expected Results", body=expected_results, prelude=expected_prelude))
    lines.extend(_render_optional_section(heading="References", body=references))
    return "\n".join(lines)


def _render_category_index(*, category: str, entries: list[ExampleDocSpec]) -> str:
    """Render one category index page as RST."""
    title = CATEGORY_TITLES[category]
    lines: list[str] = [
        title,
        "=" * len(title),
        "",
        f"Generated from canonical example docstrings/comments in ``examples/{category}``.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for entry in entries:
        lines.append(f"   {entry.slug}")
    lines.append("")
    return "\n".join(lines)


def _render_examples_index() -> str:
    """Render top-level examples index page as RST."""
    title = "Examples Guide"
    return "\n".join(
        [
            title,
            "=" * len(title),
            "",
            "The examples in this repository are runnable research-oriented scripts. They are",
            "designed to show not only API usage, but how the library fits into realistic",
            "experimental workflows. Each example lists dependencies, expected scope, and",
            "the primary concept it demonstrates.",
            "",
            "Featured Examples",
            "-----------------",
            "",
            "Direct LLM Call",
            "~~~~~~~~~~~~~~~~",
            "",
            "One-step participant execution with a configured backend client.",
            "",
            "**Requires:** base install + reachable backend endpoint",
            "**Runtime:** short",
            "**Teaches:** baseline participant setup, request execution, structured output handling",
            "",
            "Multi-Step JSON Tool Calling Agent",
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",
            "",
            "Iterative tool-using execution with explicit action/observation loops.",
            "",
            "**Requires:** base install",
            "**Runtime:** short to medium",
            "**Teaches:** tool-routing behavior, multi-step control, inspectable intermediate state",
            "",
            "Debate Pattern",
            "~~~~~~~~~~~~~~",
            "",
            "Role-based multi-agent coordination with adjudication workflow structure.",
            "",
            "**Requires:** base install",
            "**Runtime:** medium",
            "**Teaches:** orchestration patterns, delegate coordination, traceable multi-role reasoning",
            "",
            "MCP Minimal",
            "~~~~~~~~~~~",
            "",
            "Small end-to-end MCP-backed tool integration example.",
            "",
            "**Requires:** ``mcp``-compatible server/runtime setup",
            "**Runtime:** medium",
            "**Teaches:** external tool connectivity, MCP source wiring, runtime safety boundaries",
            "",
            "Deterministic runs for tests are provided by",
            "``tests/example_monkeypatch/sitecustomize.py`` when",
            "``DRA_EXAMPLE_LLM_MODE=deterministic`` is set.",
            "",
            "Full Catalog",
            "------------",
            "",
            ".. toctree::",
            "   :maxdepth: 2",
            "",
            "   agents/index",
            "   workflow/index",
            "   patterns/index",
            "   clients/index",
            "   model_selection/index",
            "   tools/index",
            "   optimization/index",
            "",
        ]
    )


def _build_specs(repo_root: Path) -> list[ExampleDocSpec]:
    """Build parsed docs specs for runnable examples."""
    specs: list[ExampleDocSpec] = []
    for path in _discover_runnable_examples(repo_root):
        rel_path = path.relative_to(repo_root).as_posix()
        rel_parts = path.relative_to(repo_root / "examples").parts
        category = rel_parts[0]
        if category not in CATEGORY_ORDER:
            continue

        if path.suffix == ".py":
            doc_text, source_start_line = _parse_python_doc_text(path)
        else:
            doc_text = _parse_shell_doc_text(path)
            source_start_line = 1

        sections = _parse_canonical_sections(
            doc_text=doc_text,
            source_path=path,
            required_sections=_required_sections_for_example(rel_path),
        )

        specs.append(
            ExampleDocSpec(
                rel_path=rel_path,
                category=category,
                slug=_slug_for_example(rel_parts=rel_parts, extension=path.suffix),
                title=_title_for_example(rel_parts=rel_parts, extension=path.suffix),
                extension=path.suffix,
                source_start_line=source_start_line,
                sections=sections,
            )
        )

    return sorted(specs, key=lambda item: (CATEGORY_ORDER.index(item.category), item.rel_path))


def _sync_file(*, path: Path, content: str, check: bool, stale: list[str]) -> None:
    """Write one generated file or record drift in check mode."""
    desired = content.rstrip() + "\n"
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == desired:
            return
    if check:
        stale.append(path.as_posix())
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(desired, encoding="utf-8")


def _sync_stale_pages(*, generated_pages: set[Path], docs_examples_root: Path, check: bool, stale: list[str]) -> None:
    """Remove stale generated pages or report drift in check mode."""
    for category in CATEGORY_ORDER:
        category_dir = docs_examples_root / category
        if not category_dir.exists():
            continue
        for existing in sorted(category_dir.glob("*.rst")):
            if existing not in generated_pages:
                if check:
                    stale.append(existing.as_posix())
                else:
                    existing.unlink()


def generate(*, repo_root: Path, check: bool) -> int:
    """Generate docs pages or validate generated pages are up to date."""
    specs = _build_specs(repo_root)
    docs_examples_root = repo_root / "docs" / "examples"

    stale: list[str] = []
    generated_pages: set[Path] = set()

    _sync_file(
        path=docs_examples_root / "index.rst",
        content=_render_examples_index(),
        check=check,
        stale=stale,
    )

    for category in CATEGORY_ORDER:
        entries = [item for item in specs if item.category == category]
        if not entries:
            continue

        category_index_path = docs_examples_root / category / "index.rst"
        generated_pages.add(category_index_path)
        _sync_file(
            path=category_index_path,
            content=_render_category_index(category=category, entries=entries),
            check=check,
            stale=stale,
        )

        for entry in entries:
            page_path = docs_examples_root / category / f"{entry.slug}.rst"
            generated_pages.add(page_path)
            _sync_file(
                path=page_path,
                content=_render_example_page(entry),
                check=check,
                stale=stale,
            )

    _sync_stale_pages(generated_pages=generated_pages, docs_examples_root=docs_examples_root, check=check, stale=stale)

    if stale:
        print("Example docs are out of date:")
        for path in sorted(stale):
            print(f"- {path}")
        return 1

    if check:
        print("Example docs are up to date.")
    else:
        print("Generated example docs.")
    return 0


def main() -> int:
    """CLI entrypoint for example docs generation/check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate generated docs are up to date.")
    args = parser.parse_args()
    return generate(repo_root=_repo_root(), check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
