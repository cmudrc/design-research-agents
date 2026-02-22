"""Example script.

Motivation
Run traced MCP runtime example for deterministic design-tool access.

Diagram
```mermaid
flowchart LR
    A["Tool input"] --> B["Tool runtime"]
    B --> C["mcp minimal result"]
    C --> D["Artifacts and trace"]
```

Technical Walkthrough
1. Configure the runtime surface for `tools` use-cases and run `mcp_minimal`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `DRA_EXAMPLE_MCP_COMMAND='python3 -m your_mcp_server_module'`
`PYTHONPATH=src python3 examples/tools/mcp_minimal.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from design_research_agents import McpServer, Toolbox, ToolResult, Tracer


def _mcp_server_command() -> tuple[str, ...]:
    raw_command = os.environ.get("DRA_EXAMPLE_MCP_COMMAND")
    if raw_command is None or not raw_command.strip():
        raise RuntimeError(
            "Set DRA_EXAMPLE_MCP_COMMAND to a stdio MCP server command "
            "(for example: 'python3 -m your_mcp_server_module')."
        )
    return tuple(shlex.split(raw_command))


def _run_report() -> dict[str, object]:
    runtime = Toolbox(
        mcp_servers=(
            McpServer(
                id="local_core",
                command=_mcp_server_command(),
                timeout_s=20,
            ),
        ),
        workspace_root=".",
        enable_core_tools=False,
    )
    try:
        mcp_tools = sorted(spec.name for spec in runtime.list_tools() if spec.name.startswith("local_core::"))
        direct_result: ToolResult = runtime.invoke(
            "local_core::text.word_count",
            {"text": "design research"},
            request_id="example-mcp-minimal",
            dependencies={},
        )
        if not direct_result.ok:
            raise RuntimeError(f"MCP tool call failed: {direct_result.error!r}")
        if not isinstance(direct_result.result, dict):
            raise RuntimeError("MCP tool call returned non-dict payload.")
        direct = direct_result.result
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
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    report = tracer.run_callable(
        agent_name="ExamplesMcpMinimal",
        request_id=request_id,
        input_payload={"scenario": "mcp-runtime-design"},
        function=_run_report,
    )
    assert isinstance(report, dict)
    report["example"] = "tools/mcp_minimal.py"
    report["trace"] = tracer.trace_info(request_id)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
