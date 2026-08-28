from __future__ import annotations

import sys
from pathlib import Path

from design_research_agents.tools import Toolbox
from design_research_agents.tools._config import CallableToolConfig, MCPServerConfig, ScriptToolConfig


def _local_mcp_server(server_id: str = "local_core") -> MCPServerConfig:
    return MCPServerConfig(
        id=server_id,
        command=(sys.executable, "-m", "design_research_agents._mcp_server"),
        env={"PYTHONPATH": "src"},
        timeout_s=20,
    )


def _rubric_script_tool() -> ScriptToolConfig:
    return ScriptToolConfig(
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


def test_mcp_server_config_python_module_uses_current_interpreter_by_default() -> None:
    config = MCPServerConfig.python_module(
        id="drp_problem",
        module="design_research_problems.mcp",
        args=("pill_capsule_min_area", "--no-citation"),
        env={"PYTHONPATH": "src"},
    )

    assert config.command == (
        sys.executable,
        "-m",
        "design_research_problems.mcp",
        "pill_capsule_min_area",
        "--no-citation",
    )
    assert config.env == {"PYTHONPATH": "src"}
    assert config.timeout_s == 20


def test_mcp_server_config_python_module_normalizes_overrides() -> None:
    config = MCPServerConfig.python_module(
        id="local",
        module=" example.server ",
        args=(Path("server-data"),),
        python=Path("/opt/python"),
        timeout_s=5,
        env_allowlist=("PATH",),
    )

    assert config.command == ("/opt/python", "-m", "example.server", "server-data")
    assert config.timeout_s == 5
    assert config.env_allowlist == ("PATH",)


def test_mcp_server_config_python_module_rejects_empty_module() -> None:
    try:
        MCPServerConfig.python_module(id="empty", module=" ")
    except ValueError as exc:
        assert "module" in str(exc)
    else:
        raise AssertionError("empty module should fail")


def test_default_constructor_lists_core_tools() -> None:
    runtime = Toolbox()

    names = {spec.name for spec in runtime.list_tools()}
    assert runtime.config.core_tools.enabled is True
    assert runtime.config.script_tools.enabled is False
    assert runtime.config.mcp.enabled is False
    assert "text.word_count" in names


def test_default_constructor_does_not_expose_network_tools() -> None:
    runtime = Toolbox()

    names = {spec.name for spec in runtime.list_tools()}
    assert runtime.config.core_tools.allow_network is False
    assert "web.instant_answer" not in names
    assert "web.search" not in names


def test_allow_network_constructor_argument_exposes_instant_answer_tool(monkeypatch) -> None:
    from design_research_agents.tools._core import _web_tools as web_tools

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    fake_payload = {
        "Heading": "Test Heading",
        "AbstractText": "Test abstract.",
        "AbstractURL": "https://example.com/test",
        "RelatedTopics": [],
    }
    monkeypatch.setattr(web_tools, "_fetch_instant_answer", lambda _query: fake_payload)

    runtime = Toolbox(allow_network=True)

    names = {spec.name for spec in runtime.list_tools()}
    assert runtime.config.core_tools.allow_network is True
    assert "web.instant_answer" in names
    assert "web.search" not in names

    result = runtime.invoke_dict(
        "web.instant_answer",
        {"query": "test query"},
        request_id="test-request",
        dependencies={},
    )
    assert result["engine"] == "duckduckgo_instant_answer"
    assert result["results"][0]["url"] == "https://example.com/test"


def test_allow_network_with_tavily_key_exposes_web_search_tool(monkeypatch) -> None:
    from design_research_agents.tools._core import _web_tools as web_tools

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    fake_tavily_payload = {
        "results": [
            {"title": "Result One", "url": "https://example.com/one", "content": "First result.", "score": 0.9},
        ]
    }
    fake_instant_answer_payload = {
        "Heading": "Test Heading",
        "AbstractText": "Test abstract.",
        "AbstractURL": "https://example.com/test",
        "RelatedTopics": [],
    }
    monkeypatch.setattr(web_tools, "_fetch_tavily_results", lambda _query, *, api_key, max_results: fake_tavily_payload)
    monkeypatch.setattr(web_tools, "_fetch_instant_answer", lambda _query: fake_instant_answer_payload)

    runtime = Toolbox(allow_network=True)

    names = {spec.name for spec in runtime.list_tools()}
    assert "web.search" in names
    assert "web.instant_answer" in names

    result = runtime.invoke_dict(
        "web.search",
        {"query": "test query"},
        request_id="test-request",
        dependencies={},
    )
    assert result["engine"] == "tavily"
    assert result["results"][0]["url"] == "https://example.com/one"

    result = runtime.invoke_dict(
        "web.instant_answer",
        {"query": "test query"},
        request_id="test-request",
        dependencies={},
    )
    assert result["engine"] == "duckduckgo_instant_answer"
    assert result["results"][0]["url"] == "https://example.com/test"


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
            CallableToolConfig(
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
