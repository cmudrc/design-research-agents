## Lazy Tool Examples

This folder shows two ways to use lazy tools:
- Direct script execution (stdin JSON in, stdout JSON envelope out).
- One-step agent execution (model selects and invokes the lazy tool).

Use these examples to verify:
- Lazy tool headers are discovered correctly.
- Lazy tool runtime execution and artifact writing work end-to-end.
- Agent tool selection can target lazy tools (`lazy::<tool_name>`).

## What Each Example Does

- `examples/lazy_tools/python/rubric_score.py`
  - A lazy tool that scores input text with a small rubric and writes `artifacts/rubric_score/rubric_score_report.json`.
- `examples/lazy_tools/bash/repo_quickscan.sh`
  - A lazy tool that produces a quick repository listing and writes `artifacts/repo_quickscan/report.txt`.
- `examples/lazy_tools/python/single_step_json_lazy_rubric_score_agent.py`
  - Runs `SingleStepJsonToolCallingAgent` against `lazy::rubric_score`.
- `examples/lazy_tools/bash/single_step_json_lazy_repo_quickscan_agent.py`
  - Runs `SingleStepJsonToolCallingAgent` against `lazy::repo_quickscan`.

## Quick Start

Run from repository root:

```bash
# 1) Direct lazy tool: Python
PYTHONPATH=src python3 examples/lazy_tools/python/rubric_score.py <<'JSON'
{"text":"one two three four five","max_score":10}
JSON

# 2) Direct lazy tool: Bash
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
  - `output.tool_name` is the expected lazy tool name.
  - `tool_results[0].ok` is `true`.
  - `tool_results[0].artifacts` includes the expected artifact path.

## Notes

- Agent wrappers use `LlamaCppServerLLMClient()`.
- If your local llama-cpp server is cold, first run may take longer due to startup.
