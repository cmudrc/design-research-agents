## Optimization Examples

This folder contains traced optimization-oriented agent runs tied to
engineering-design interpretation.

## Scripts

- `multi_step_json_tool_calling_1d_optimization.py`
  - `MultiStepAgent(mode="json")` minimizing `f(x)=x^2` with LLM-selected tools.

## Quick Start

```bash
PYTHONPATH=src python3 examples/optimization/multi_step_json_tool_calling_1d_optimization.py
```

## Expected Outputs

- JSON envelope with objective history, best-seen point, and final step metadata.
- Trace metadata pointing to JSONL run artifact.
