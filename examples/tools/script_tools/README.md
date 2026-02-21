## Script Tool Examples

This folder demonstrates script-tool execution with design-oriented outputs and
trace artifact emission.

## Scripts

- `examples/tools/script_tools/python/rubric_score.py`
  - Scores input text against a compact rubric and writes JSON report + trace artifact.
- `examples/tools/script_tools/bash/repo_quickscan.sh`
  - Produces repository inventory report and emits a trace artifact.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/tools/script_tools/python/rubric_score.py <<'JSON'
{"text":"design review notes for quick-release latch and gasket alignment","max_score":10}
JSON

bash examples/tools/script_tools/bash/repo_quickscan.sh <<'JSON'
{"include_hidden":false}
JSON
```

## Expected Outputs

- One JSON envelope with `ok`, `result`, `artifacts`, `warnings`, `error`.
- `result.trace_path` and corresponding trace artifact in `artifacts/examples/traces`.
