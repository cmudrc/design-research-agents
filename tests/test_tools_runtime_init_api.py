from __future__ import annotations

import sys
from pathlib import Path

from design_research_agents.tools import UnifiedToolRuntime
from design_research_agents.tools.config import McpServerConfig


def _local_mcp_server(server_id: str = "local_core") -> McpServerConfig:
    return McpServerConfig(
        id=server_id,
        command=(sys.executable, "-m", "design_research_agents.mcp_server"),
        env={"PYTHONPATH": "src"},
        timeout_s=20,
    )


def test_default_constructor_lists_core_tools() -> None:
    runtime = UnifiedToolRuntime()

    names = {spec.name for spec in runtime.list_tools()}
    assert runtime.config.core_tools.enabled is True
    assert runtime.config.lazy_tools.enabled is False
    assert runtime.config.mcp.enabled is False
    assert "calculator" in names


def test_constructor_enables_lazy_tools_from_search_paths() -> None:
    runtime = UnifiedToolRuntime(
        workspace_root=".",
        enable_core_tools=False,
        lazy_search_paths=("examples/lazy_tools",),
    )

    names = {spec.name for spec in runtime.list_tools()}
    assert runtime.config.core_tools.enabled is False
    assert runtime.config.lazy_tools.enabled is True
    assert "lazy::rubric_score" in names


def test_constructor_enables_mcp_from_servers() -> None:
    runtime = UnifiedToolRuntime(mcp_servers=(_local_mcp_server(),))
    try:
        assert runtime.config.mcp.enabled is True
        assert tuple(server.id for server in runtime.config.mcp.servers) == ("local_core",)
    finally:
        runtime.close()


def test_lazy_and_mcp_classmethods_disable_core_tools_by_default() -> None:
    lazy_runtime = UnifiedToolRuntime.lazy(search_paths=("examples/lazy_tools",))
    mcp_runtime = UnifiedToolRuntime.mcp(servers=(_local_mcp_server(),))
    try:
        assert lazy_runtime.config.core_tools.enabled is False
        assert mcp_runtime.config.core_tools.enabled is False
    finally:
        mcp_runtime.close()


def test_pathlike_workspace_root_is_normalized_and_runtime_still_invokes() -> None:
    runtime = UnifiedToolRuntime(workspace_root=Path("."))

    assert isinstance(runtime.config.core_tools.workspace_root, str)
    result = runtime.invoke(
        "calculator",
        {"expression": "2 + 2"},
        request_id="init-api",
        dependencies={},
    )
    assert result.ok is True
