from __future__ import annotations

import sys
from pathlib import Path

from design_research_agents.tools import Toolbox
from design_research_agents.tools._config import CallableTool, McpServer, ScriptTool


def _local_mcp_server(server_id: str = "local_core") -> McpServer:
    return McpServer(
        id=server_id,
        command=(sys.executable, "-m", "design_research_agents._mcp_server"),
        env={"PYTHONPATH": "src"},
        timeout_s=20,
    )


def _rubric_script_tool() -> ScriptTool:
    return ScriptTool(
        name="rubric_score",
        path="examples/tools/script_tools/rubric_score.py",
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


def test_default_constructor_lists_core_tools() -> None:
    runtime = Toolbox()

    names = {spec.name for spec in runtime.list_tools()}
    assert runtime.config.core_tools.enabled is True
    assert runtime.config.script_tools.enabled is False
    assert runtime.config.mcp.enabled is False
    assert "text.word_count" in names


def test_constructor_enables_script_tools() -> None:
    runtime = Toolbox(
        workspace_root=".",
        enable_core_tools=False,
        script_tools=(_rubric_script_tool(),),
    )

    names = {spec.name for spec in runtime.list_tools()}
    assert runtime.config.core_tools.enabled is False
    assert runtime.config.script_tools.enabled is True
    assert "script::rubric_score" in names


def test_constructor_enables_mcp_from_servers() -> None:
    runtime = Toolbox(mcp_servers=(_local_mcp_server(),))
    try:
        assert runtime.config.mcp.enabled is True
        assert tuple(server.id for server in runtime.config.mcp.servers) == ("local_core",)
    finally:
        runtime.close()


def test_constructor_registers_callable_tools() -> None:
    runtime = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="echo.callable",
                description="Echo the payload",
                handler=lambda payload: {"payload": dict(payload)},
            ),
        ),
    )
    result = runtime.invoke(
        "echo.callable",
        {"x": 1},
        request_id="init-api",
        dependencies={},
    )
    assert result.ok is True
    assert result.result == {"payload": {"x": 1}}


def test_pathlike_workspace_root_is_normalized_and_runtime_still_invokes() -> None:
    runtime = Toolbox(workspace_root=Path("."))

    assert isinstance(runtime.config.core_tools.workspace_root, str)
    result = runtime.invoke(
        "text.word_count",
        {"text": "one two three"},
        request_id="init-api",
        dependencies={},
    )
    assert result.ok is True
    assert isinstance(result.result, dict)
    assert result.result["word_count"] == 3
