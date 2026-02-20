## Optimization Examples

This folder demonstrates optimization-oriented agent runs with callable tools.

## What Each Example Demonstrates

- `multi_step_tool_router_1d_optimization.py`
  - `MultiStepAgent(mode="json")` router special-case with separate
    `optimizer.increase_x` and `optimizer.decrease_x` tools to minimize `f(x)=x^2`.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/optimization/multi_step_tool_router_1d_optimization.py
```

## Expected Outputs

- The script prints a JSON envelope.
- The multi-step example includes `step_outputs`, stop/final metadata, and
  controller diagnostics (`terminated_reason`, non-improving-step detection,
  `best_seen`, and a `memory_tail` snapshot).
