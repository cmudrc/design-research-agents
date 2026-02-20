## Script Tool Examples

This folder shows two ways to use script tools:
- Direct script execution (stdin JSON in, stdout JSON envelope out).

Use these examples to verify:
- Script tool execution and artifact writing work end-to-end.

## What Each Example Does

- `examples/tools/script_tools/python/rubric_score.py`
  - A script tool that scores input text with a small rubric and writes `artifacts/rubric_score/rubric_score_report.json`.
- `examples/tools/script_tools/bash/repo_quickscan.sh`
  - A script tool that produces a quick repository listing and writes `artifacts/repo_quickscan/report.txt`.

## Quick Start

Run from repository root:

```bash
# 1) Direct script tool: Python
PYTHONPATH=src python3 examples/tools/script_tools/python/rubric_score.py <<'JSON'
{"text":"one two three four five","max_score":10}
JSON

# 2) Direct script tool: Bash
bash examples/tools/script_tools/bash/repo_quickscan.sh <<'JSON'
{"include_hidden":false}
JSON
```

## Expected Outputs

- Direct tools print one JSON envelope with `ok`, `result`, `artifacts`, `warnings`, `error`.

## Notes

- If your local llama-cpp server is cold, first run may take longer due to startup.
