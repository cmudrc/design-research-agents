from __future__ import annotations

from pathlib import Path

from design_research_agents.tools import UnifiedToolRuntime
from design_research_agents.tools.lazy.discovery import discover_lazy_tools
from design_research_agents.tools.lazy.parser import LazyHeaderError, parse_lazy_tool_header


def test_lazy_parser_python_and_bash_headers_parse() -> None:
    py_header = parse_lazy_tool_header("examples/lazy_tools/python/rubric_score.py")
    sh_header = parse_lazy_tool_header("examples/lazy_tools/bash/repo_quickscan.sh")

    assert py_header.tool_name == "rubric_score"
    assert py_header.outputs_stdout_json is True
    assert sh_header.tool_name == "repo_quickscan"


def test_lazy_parser_reports_directive_errors_with_line_numbers(tmp_path: Path) -> None:
    script = tmp_path / "broken_tool.py"
    script.write_text(
        "\n".join(
            [
                '"""',
                "@description: missing tool name",
                "@inputs:",
                "  bad_line",
                '"""',
                "print('x')",
            ]
        ),
        encoding="utf-8",
    )

    try:
        parse_lazy_tool_header(script)
    except LazyHeaderError as exc:
        message = str(exc)
        assert "Missing @tool_name" in message or "line" in message
    else:
        raise AssertionError("Expected LazyHeaderError for malformed header.")


def test_lazy_discovery_ignores_non_lazy_scripts(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text(
        "\n".join(
            [
                '"""Regular helper script, not a lazy tool."""',
                "print('hello')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    tools, diagnostics = discover_lazy_tools((str(tmp_path),))
    assert not tools
    assert not diagnostics


def test_lazy_discovery_and_run_examples() -> None:
    tools, diagnostics = discover_lazy_tools(("examples/lazy_tools",))
    names = {tool.header.tool_name for tool in tools}

    assert "rubric_score" in names
    assert "repo_quickscan" in names
    assert not diagnostics

    runtime = UnifiedToolRuntime(
        workspace_root=".",
        enable_core_tools=False,
        lazy_search_paths=("examples/lazy_tools",),
    )

    result = runtime.invoke(
        "lazy::rubric_score",
        {"text": "one two three four five six seven eight nine ten"},
        request_id="unit-test",
        dependencies={},
    )
    assert result.ok is True
    assert isinstance(result.result, dict)
    assert "score" in result.result
    assert result.artifacts
