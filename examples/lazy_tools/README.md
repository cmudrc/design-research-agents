## Script Tool Examples

This folder shows two ways to use script tools:
- Direct script execution (stdin JSON in, stdout JSON envelope out).
- One-step agent execution (model selects and invokes the script tool).

Use these examples to verify:
- Script tool execution and artifact writing work end-to-end.
- Agent tool selection can target script tools (`script::<tool_name>`).

## What Each Example Does

- `examples/lazy_tools/python/rubric_score.py`
  - A script tool that scores input text with a small rubric and writes `artifacts/rubric_score/rubric_score_report.json`.
- `examples/lazy_tools/bash/repo_quickscan.sh`
  - A script tool that produces a quick repository listing and writes `artifacts/repo_quickscan/report.txt`.
- `examples/lazy_tools/python/single_step_json_lazy_rubric_score_agent.py`
  - Runs `SingleStepJsonToolCallingAgent` against `script::rubric_score`.
- `examples/lazy_tools/bash/single_step_json_lazy_repo_quickscan_agent.py`
  - Runs `SingleStepJsonToolCallingAgent` against `script::repo_quickscan`.

## Quick Start

Run from repository root:

```bash
# 1) Direct script tool: Python
PYTHONPATH=src python3 examples/lazy_tools/python/rubric_score.py <<'JSON'
{"text":"one two three four five","max_score":10}
JSON

# 2) Direct script tool: Bash
bash examples/lazy_tools/bash/repo_quickscan.sh <<'JSON'
{"include_hidden":false}
JSON

# 3) Agent wrapper: rubric_score
PYTHONPATH=src python3 examples/lazy_tools/python/single_step_json_lazy_rubric_score_agent.py

# 4) Agent wrapper: repo_quickscan
PYTHONPATH=src python3 examples/lazy_tools/bash/single_step_json_lazy_repo_quickscan_agent.py
```

## Expected Outputs

- Direct tools print one JSON envelope with `ok`, `result`, `artifacts`, `warnings`, `error`.
- Agent wrappers print an `AgentResult` where:
  - `output.tool_name` is the expected script tool name.
  - `tool_results[0].ok` is `true`.
  - `tool_results[0].artifacts` includes the expected artifact path.

## Notes

- Agent wrappers use `LlamaCppServerLLMClient()`.
- If your local llama-cpp server is cold, first run may take longer due to startup.
