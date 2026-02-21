## Design-Focused Examples

This directory contains runnable engineering-design examples for every major
public surface in `design_research_agents`.

All runnable examples now:
- use intentional design/engineering scenarios,
- emit structured output,
- include trace metadata pointing to JSONL artifacts in `artifacts/examples/traces`.

## Directory Guide

- `examples/agents`
  - Traced direct and multi-step agent execution patterns.
- `examples/clients`
  - Traced client configuration snapshots for each LLM client class.
- `examples/workflow`
  - Traced orchestration/pattern examples including full workflow step coverage.
- `examples/model_selection`
  - Traced local-vs-remote policy decisions.
- `examples/optimization`
  - Traced optimization loop built on JSON router-special-case behavior.
- `examples/tools`
  - Traced tool runtime examples across core/script/MCP sources.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/agents/basic/direct_llm_call.py
```

## Expected Outputs

- Scripts print a JSON payload (except `workflow_runtime.py`, which prints a Python dict for smoke-test compatibility).
- Output includes a `trace` block with request id and trace path.
- Script-tool examples emit tool envelopes and include trace artifact references.
