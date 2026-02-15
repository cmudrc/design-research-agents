"""Minimal MCP runtime example with deterministic local stdio server."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping

from design_research_agents import Toolbox
from design_research_agents.tools.config import McpServer


def _invoke_dict(
    runtime: Toolbox,
    tool_name: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    result = runtime.invoke(tool_name, payload, request_id="example-mcp-minimal", dependencies={})
    if not result.ok:
        message = result.error.message if result.error is not None else "unknown tool error"
        raise RuntimeError(f"{tool_name} failed: {message}")
    if not isinstance(result.result, dict):
        raise RuntimeError(f"{tool_name} returned non-object payload.")
    return result.result


def main() -> None:
    """Run a minimal MCP-only runtime and print one compact report."""
    runtime = Toolbox(
        mcp_servers=(
            McpServer(
                id="local_core",
                command=(sys.executable, "-m", "design_research_agents.mcp_server"),
                env={"PYTHONPATH": "src"},
                timeout_s=20,
            ),
        ),
        workspace_root=".",
        enable_core_tools=False,
    )

    try:
        mcp_tools = sorted(
            spec.name for spec in runtime.list_tools() if spec.name.startswith("local_core::")
        )
        direct = _invoke_dict(runtime, "local_core::calculator", {"expression": "(9 + 3) / 2"})
    finally:
        runtime.close()

    print(
        json.dumps(
            {
                "mcp_tool_count": len(mcp_tools),
                "sample_tools": mcp_tools[:5],
                "direct_result": direct["result"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
