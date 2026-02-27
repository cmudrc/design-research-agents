r"""# Tools / MCP Minimal.

## Introduction
MCP and JSON-RPC define interoperable tool transport contracts, and Toolformer motivates why model behavior
improves when tool calls are explicit and machine-checked. This example provides the minimal MCP
server/toolbox path for validating protocol-level integration inside the framework runtime.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Toolbox.invoke(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["Toolbox.invoke(...)"]
    C --> D["MCP server registration exposes namespaced tool contracts"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "direct_result": 2,
     "example": "tools/mcp_minimal.py",
     "mcp_tool_count": 23,
     "sample_tools": [
       "local_core::bash.exec",
       "local_core::data.describe",
       "local_core::data.load_csv",
       "local_core::eval.decision_matrix",
       "local_core::eval.pairwise_rank"
     ],
     "trace": {
       "request_id": "example-tools-mcp-minimal-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-tools-mcp-minimal-design-001.jsonl"
     }
   }


## References
- `Model Context Protocol Specification <https://modelcontextprotocol.io/specification/2025-06-18>`_
- `JSON-RPC 2.0 Specification <https://www.jsonrpc.org/specification>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from design_research_agents import MCPServerConfig, Toolbox, ToolResult, Tracer


def _run_report() -> dict[str, object]:
    runtime = Toolbox(
        mcp_servers=(
            MCPServerConfig(
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
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        runtime.close()

    return {
        "mcp_tool_count": len(mcp_tools),
        "sample_tools": mcp_tools[:5],
        "direct_result": direct["word_count"],
    }


def main() -> None:
    """Run traced MCP report generation and print JSON result."""
    # Fixed request id keeps traces and docs output deterministic across runs.
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
