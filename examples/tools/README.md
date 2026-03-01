## Tool Runtime Examples

These examples exercise traced `Toolbox` behavior across core, script, and MCP
sources for design-analysis workflows.

## Scripts

- `mcp_minimal.py`
  - Traced MCP-only runtime and namespaced invocation.
- `multi_source_tool_usage.py`
  - Traced multi-source run combining core/script/MCP tools.
- `script_tools/README.md`
  - Script-tool examples and direct execution commands.

## Quick Start

```bash
PYTHONPATH=src python3 examples/tools/mcp_minimal.py
PYTHONPATH=src python3 examples/tools/multi_source_tool_usage.py
bash examples/tools/script_tools/repo_quickscan.sh <<'JSON'
{"include_hidden":false}
JSON
```

## Expected Outputs

- Compact JSON report payloads.
- Artifacts written under `artifacts/examples` and script-tool artifact folders.
- Trace metadata in each output payload.
