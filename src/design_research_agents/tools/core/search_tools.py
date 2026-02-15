"""Search tools for local workspace text lookup."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from shutil import which

from design_research_agents.contracts.tools import ToolMetadata, ToolSideEffects, ToolSpec
from design_research_agents.tools.policy import ToolPolicy
from design_research_agents.tools.sources.inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str


def register_search_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register workspace text-search tools with rg and Python fallback."""
    source.register_tool(
        spec=ToolSpec(
            name="search.ripgrep",
            description=(
                "Search for a text pattern in the workspace using ripgrep when available "
                "and a Python fallback otherwise."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "root": {"type": "string"},
                    "globs": {"type": "array", "items": {"type": "string"}},
                    "max_matches": {"type": "integer"},
                    "context_lines": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=ToolMetadata(
                source="core",
                side_effects=ToolSideEffects(filesystem_read=True, commands=("rg",)),
                timeout_s=20,
                max_output_bytes=65_536,
                risky=True,
            ),
        ),
        handler=lambda i, r, d: _search(i, policy=policy),
    )


def _search(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    query = get_str(input_dict, "query").strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    root_path = policy.resolve_read_path(get_str(input_dict, "root", default="."))
    max_matches = get_int(input_dict, "max_matches", default=200)
    context_lines = get_int(input_dict, "context_lines", default=0)
    globs_raw = input_dict.get("globs")
    globs: list[str] = []
    if isinstance(globs_raw, list):
        globs = [str(item).strip() for item in globs_raw if str(item).strip()]

    rg_binary = which("rg")
    if rg_binary:
        return _search_with_rg(
            rg_binary=rg_binary,
            root=root_path,
            query=query,
            globs=globs,
            max_matches=max_matches,
            context_lines=context_lines,
        )

    return _search_with_python(
        root=root_path,
        query=query,
        max_matches=max_matches,
    )


def _search_with_rg(
    *,
    rg_binary: str,
    root: Path,
    query: str,
    globs: list[str],
    max_matches: int,
    context_lines: int,
) -> Mapping[str, object]:
    command = [
        rg_binary,
        "--line-number",
        "--column",
        "--no-heading",
        "--color",
        "never",
        "--max-count",
        str(max_matches),
    ]
    if context_lines > 0:
        command.extend(["-C", str(context_lines)])
    for glob in globs:
        command.extend(["-g", glob])
    command.extend([query, str(root)])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    matches: list[dict[str, object]] = []
    line_pattern = re.compile(r"^(.*?):(\d+):(\d+):(.*)$")
    for raw_line in completed.stdout.splitlines():
        parsed = line_pattern.match(raw_line)
        if parsed is None:
            continue
        file_path, line_no, column_no, snippet = parsed.groups()
        line_int = int(line_no)
        column_int = int(column_no)
        matches.append(
            {
                "file": file_path,
                "line": line_int,
                "column": column_int,
                "snippet": snippet,
                "ref": f"{file_path}:{line_int}:{column_int}",
            }
        )
        if len(matches) >= max_matches:
            break

    return {
        "engine": "rg",
        "query": query,
        "root": str(root),
        "count": len(matches),
        "matches": matches,
        "stderr": completed.stderr.strip(),
    }


def _search_with_python(
    *,
    root: Path,
    query: str,
    max_matches: int,
) -> Mapping[str, object]:
    pattern = re.compile(query)
    matches: list[dict[str, object]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            match = pattern.search(line)
            if match is None:
                continue
            column = match.start() + 1
            matches.append(
                {
                    "file": str(path),
                    "line": line_no,
                    "column": column,
                    "snippet": line,
                    "ref": f"{path}:{line_no}:{column}",
                }
            )
            if len(matches) >= max_matches:
                return {
                    "engine": "python",
                    "query": query,
                    "root": str(root),
                    "count": len(matches),
                    "matches": matches,
                }

    return {
        "engine": "python",
        "query": query,
        "root": str(root),
        "count": len(matches),
        "matches": matches,
    }
