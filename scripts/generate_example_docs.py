#!/usr/bin/env python3
"""Generate per-example Sphinx pages from canonical example doc sections."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

REQUIRED_SECTIONS = (
    "Motivation",
    "Diagram",
    "Technical Walkthrough",
    "Expected Results",
    "Discussion",
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


def _parse_python_doc_text(path: Path) -> str:
    """Parse module docstring text from one Python example."""
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    docstring = ast.get_docstring(module, clean=False)
    if not isinstance(docstring, str) or not docstring.strip():
        raise ValueError(f"{path}: missing module docstring.")
    return docstring


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


def _parse_canonical_sections(*, doc_text: str, source_path: Path) -> dict[str, str]:
    """Parse canonical docs sections from one source doc block."""
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in doc_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped in REQUIRED_SECTIONS:
            current_section = stripped
            sections[current_section] = []
            continue
        if current_section is not None:
            sections[current_section].append(line)

    missing = [section for section in REQUIRED_SECTIONS if section not in sections]
    if missing:
        raise ValueError(f"{source_path}: missing canonical section(s): {missing}")

    return {name: "\n".join(sections[name]).strip() for name in REQUIRED_SECTIONS}


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


def _extract_mermaid(diagram_section: str, *, source_path: str) -> str:
    """Extract Mermaid diagram text from canonical Diagram section."""
    lines = diagram_section.splitlines()
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
        mermaid_text = diagram_section.strip()

    if not mermaid_text:
        raise ValueError(f"{source_path}: Diagram section must include Mermaid content.")
    return mermaid_text


def _render_example_page(spec: ExampleDocSpec) -> str:
    """Render one example page as RST."""
    title = spec.title
    run_command = f"PYTHONPATH=src python3 {spec.rel_path}" if spec.extension == ".py" else f"bash {spec.rel_path}"
    include_path = f"../../../{spec.rel_path}"
    literal_language = "python" if spec.extension == ".py" else "bash"

    motivation = spec.sections["Motivation"]
    walkthrough = spec.sections["Technical Walkthrough"]
    expected_results = spec.sections["Expected Results"]
    discussion = spec.sections["Discussion"]
    mermaid = _extract_mermaid(spec.sections["Diagram"], source_path=spec.rel_path)
    indented_mermaid = "\n".join(f"   {line}" for line in mermaid.splitlines())

    return "\n".join(
        [
            title,
            "=" * len(title),
            "",
            f"Source: ``{spec.rel_path}``",
            "",
            "Run Command",
            "-----------",
            "",
            ".. code-block:: bash",
            "",
            f"   {run_command}",
            "",
            "Motivation",
            "----------",
            "",
            motivation,
            "",
            "Diagram",
            "-------",
            "",
            ".. mermaid::",
            "",
            indented_mermaid,
            "",
            "Technical Walkthrough",
            "---------------------",
            "",
            walkthrough,
            "",
            "Expected Results",
            "----------------",
            "",
            expected_results,
            "",
            "Discussion",
            "----------",
            "",
            discussion,
            "",
            "Source Code",
            "-----------",
            "",
            f".. literalinclude:: {include_path}",
            f"   :language: {literal_language}",
            "   :linenos:",
            "",
        ]
    )


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
            "Per-example documentation is generated from runnable example docstrings/comments.",
            "",
            "Deterministic runs for tests are provided by",
            "``tests/example_monkeypatch/sitecustomize.py`` when",
            "``DRA_EXAMPLE_LLM_MODE=deterministic`` is set.",
            "",
            "Categories",
            "----------",
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

        doc_text = _parse_python_doc_text(path) if path.suffix == ".py" else _parse_shell_doc_text(path)
        sections = _parse_canonical_sections(doc_text=doc_text, source_path=path)

        specs.append(
            ExampleDocSpec(
                rel_path=rel_path,
                category=category,
                slug=_slug_for_example(rel_parts=rel_parts, extension=path.suffix),
                title=_title_for_example(rel_parts=rel_parts, extension=path.suffix),
                extension=path.suffix,
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
