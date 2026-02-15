from __future__ import annotations

import sys

from design_research_agents.tools import ToolRuntimeConfig, UnifiedToolRuntime
from design_research_agents.tools.config import CoreToolsConfig, McpConfig, McpServerConfig


def test_mcp_stdio_server_list_and_call() -> None:
    config = ToolRuntimeConfig(
        core_tools=CoreToolsConfig(enabled=False, workspace_root="."),
        mcp=McpConfig(
            enabled=True,
            servers=(
                McpServerConfig(
                    id="local_core",
                    command=(sys.executable, "-m", "design_research_agents.mcp_server"),
                    env={"PYTHONPATH": "src"},
                    timeout_s=20,
                ),
            ),
        ),
    )
    runtime = UnifiedToolRuntime(config=config)
    try:
        names = {spec.name for spec in runtime.list_tools()}
        assert "local_core::calculator" in names

        qualified = runtime.invoke(
            "local_core::calculator",
            {"expression": "2 + 3"},
            request_id="unit-test",
            dependencies={},
        )
        assert qualified.ok is True
        assert isinstance(qualified.result, dict)
        assert qualified.result["result"] == 5.0

        unqualified = runtime.invoke(
            "calculator",
            {"expression": "8 - 3"},
            request_id="unit-test",
            dependencies={},
        )
        assert unqualified.ok is True
        assert isinstance(unqualified.result, dict)
        assert unqualified.result["result"] == 5.0
    finally:
        runtime.close()
