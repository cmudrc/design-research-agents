from __future__ import annotations

import sys

from design_research_agents.tools import Toolbox
from design_research_agents.tools._config import McpServer


def test_mcp_stdio_server_list_and_call() -> None:
    runtime = Toolbox(
        mcp_servers=(
            McpServer(
                id="local_core",
                command=(sys.executable, "-m", "design_research_agents._mcp_server"),
                env={"PYTHONPATH": "src"},
                timeout_s=20,
            ),
        ),
        workspace_root=".",
        enable_core_tools=False,
    )
    try:
        names = {spec.name for spec in runtime.list_tools()}
        assert "local_core::text.word_count" in names

        qualified = runtime.invoke(
            "local_core::text.word_count",
            {"text": "design research"},
            request_id="unit-test",
            dependencies={},
        )
        assert qualified.ok is True
        assert isinstance(qualified.result, dict)
        assert qualified.result["word_count"] == 2

        unqualified = runtime.invoke(
            "text.word_count",
            {"text": "design research"},
            request_id="unit-test",
            dependencies={},
        )
        assert unqualified.ok is False
        assert unqualified.error is not None
        assert "not registered" in unqualified.error.message
    finally:
        runtime.close()
