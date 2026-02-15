from __future__ import annotations

from pathlib import Path

from design_research_agents.tools import Toolbox
from design_research_agents.tools.config import ScriptTool


def _rubric_script() -> ScriptTool:
    return ScriptTool(
        name="rubric_score",
        path="examples/lazy_tools/python/rubric_score.py",
        description="Score text against a simple rubric.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "max_score": {"type": "integer"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        filesystem_write=True,
    )


def _quickscan_script() -> ScriptTool:
    return ScriptTool(
        name="repo_quickscan",
        path="examples/lazy_tools/bash/repo_quickscan.sh",
        description="Produce a quick repository inventory snapshot.",
        input_schema={
            "type": "object",
            "properties": {
                "include_hidden": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        filesystem_read=True,
        filesystem_write=True,
    )


def test_script_tools_are_listed_and_runnable() -> None:
    runtime = Toolbox(
        workspace_root=".",
        enable_core_tools=False,
        script_tools=(_rubric_script(), _quickscan_script()),
    )

    names = {spec.name for spec in runtime.list_tools()}
    assert names == {"script::repo_quickscan", "script::rubric_score"}

    result = runtime.invoke(
        "script::rubric_score",
        {"text": "one two three four five six seven eight nine ten"},
        request_id="unit-test",
        dependencies={},
    )
    assert result.ok is True
    assert isinstance(result.result, dict)
    assert "score" in result.result
    assert result.artifacts


def test_script_lint_target_can_be_used_with_example_path(tmp_path: Path) -> None:
    helper = tmp_path / "helper.py"
    helper.write_text("print('hello')\n", encoding="utf-8")

    runtime = Toolbox(enable_core_tools=False, script_tools=(_rubric_script(),))
    names = {spec.name for spec in runtime.list_tools()}
    assert "script::rubric_score" in names
