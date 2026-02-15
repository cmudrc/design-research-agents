## Tool Runtime Examples

These examples focus on `UnifiedToolRuntime` behavior across different tool
sources.

## What Each Example Demonstrates

- `mcp_minimal.py`
  - Minimal MCP-only runtime with canonical namespaced tool calls.
- `source_fusion_story.py`
  - One run combining core tools, lazy tools, and MCP tools into a single report.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/tools/mcp_minimal.py
PYTHONPATH=src python3 examples/tools/source_fusion_story.py
```

## Expected Outputs

- `mcp_minimal.py` prints a compact JSON report with MCP tool inventory and sample results.
- `source_fusion_story.py` prints a combined report and writes artifacts under
  `artifacts/examples`.

## Troubleshooting

- MCP startup failures:
  - Confirm `PYTHONPATH=src` and that MCP dependencies are installed.
- Missing lazy tools in `source_fusion_story.py`:
  - Verify `examples/lazy_tools` exists and lazy headers are valid.
