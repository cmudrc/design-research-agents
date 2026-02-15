## Tool Runtime Examples

These examples focus on `Toolbox` behavior across different tool
sources.

## What Each Example Demonstrates

- `mcp_minimal.py`
  - Minimal MCP-only runtime with canonical namespaced tool calls.
- `source_fusion_story.py`
  - One run combining core tools, script tools, and MCP tools into a single report.
- `script_tools/README.md`
  - Script-tool examples and one-step agent wrappers, colocated under `examples/tools/script_tools`.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/tools/mcp_minimal.py
PYTHONPATH=src python3 examples/tools/source_fusion_story.py
PYTHONPATH=src python3 examples/tools/script_tools/python/single_step_json_script_rubric_score_agent.py
bash examples/tools/script_tools/bash/repo_quickscan.sh <<'JSON'
{"include_hidden":false}
JSON
```

## Expected Outputs

- `mcp_minimal.py` prints a compact JSON report with MCP tool inventory and sample results.
- `source_fusion_story.py` prints a combined report and writes artifacts under
  `artifacts/examples`.
- Script-tool examples print one JSON envelope and write artifacts under
  `artifacts/repo_quickscan` or `artifacts/rubric_score`.

## Troubleshooting

- MCP startup failures:
  - Confirm `PYTHONPATH=src` and that MCP dependencies are installed.
- Missing script tools in `source_fusion_story.py`:
  - Verify `examples/tools/script_tools` exists and script tool paths are valid.
