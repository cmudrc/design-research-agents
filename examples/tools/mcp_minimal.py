"""Run traced MCP runtime example for deterministic design-tool access.

Expected observations:
- ``mcp_tool_count`` is non-zero.
- ``direct_result`` contains text metric output.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

from design_research_agents import McpServer, Toolbox
from design_research_agents._shared._example_support import (
    print_json,
    run_traced_callable,
    trace_info,
)


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


def _run_report() -> dict[str, object]:
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
        mcp_tools = sorted(spec.name for spec in runtime.list_tools() if spec.name.startswith("local_core::"))
        direct = _invoke_dict(runtime, "local_core::text.word_count", {"text": "design research"})
    finally:
        runtime.close()

    return {
        "mcp_tool_count": len(mcp_tools),
        "sample_tools": mcp_tools[:5],
        "direct_result": direct["word_count"],
    }


def main() -> None:
    """Run traced MCP report generation and print JSON result."""
    request_id = "example-tools-mcp-minimal-design-001"
    report = run_traced_callable(
        agent_name="ExamplesMcpMinimal",
        request_id=request_id,
        input_payload={"scenario": "mcp-runtime-design"},
        function=_run_report,
    )
    assert isinstance(report, dict)
    report["example"] = "tools/mcp_minimal.py"
    report["trace"] = trace_info(request_id)
    print_json(report)


if __name__ == "__main__":
    main()
