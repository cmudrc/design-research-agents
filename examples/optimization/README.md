## Optimization Examples

This folder demonstrates optimization-oriented agent runs with callable tools.

## What Each Example Demonstrates

- `multi_step_tool_router_1d_optimization.py`
  - Multi-step ToolRouting with separate `optimizer.increase_x` and
    `optimizer.decrease_x` tools to minimize `f(x)=x^2`.
- `single_step_optimizer_tool_agent.py`
  - Single-step JSON tool-calling agent that launches a dedicated
    `optimizer.search_1d` tool from an initial guess.
  - Uses `scipy.optimize.minimize` when SciPy is installed.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/optimization/multi_step_tool_router_1d_optimization.py
PYTHONPATH=src python3 examples/optimization/single_step_optimizer_tool_agent.py
```

## Expected Outputs

- Both scripts print a JSON envelope.
- The multi-step example includes `step_outputs`, stop/final metadata, and
  controller diagnostics (`terminated_reason`, non-improving-step detection,
  `best_seen`, and a `memory_tail` snapshot).
- The single-step example includes selected tool input plus optimization history.
