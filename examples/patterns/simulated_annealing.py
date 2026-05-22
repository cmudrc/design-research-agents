"""# Patterns / Simulated Annealing.

## Introduction
Simulated annealing is useful when a design space has a numeric objective,
lightweight constraints, and local moves that may temporarily get worse before
finding a better basin. This example keeps the delegates deterministic so the
runtime contract is easy to inspect without an LLM dependency.


## Technical Implementation
1. Define a local objective delegate for a one-dimensional quadratic target.
2. Define a neighbor delegate that proposes bounded local moves.
3. Execute ``SimulatedAnnealingPattern.run(...)`` through the public patterns API.
4. Print a compact JSON payload for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Initial state"] --> B["SimulatedAnnealingPattern.run(...)"]
    B --> C["neighbor_delegate proposes local moves"]
    C --> D["objective_delegate scores each state"]
    D --> E["Metropolis acceptance + convergence checks"]
    E --> F["ExecutionResult/payload"]
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape:

.. code-block:: text

   {
     "best_objective_value": 0.0,
     "best_state": {
       "x": 3.0
     },
     "iterations": 6,
     "success": true,
     "terminated_reason": "max_iterations_reached"
   }

## References
- `Simulated Annealing <https://en.wikipedia.org/wiki/Simulated_annealing>`_
- `Metropolis-Hastings Algorithm <https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`_
- `Mathematical Optimization <https://en.wikipedia.org/wiki/Mathematical_optimization>`_
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents import (
    AdaptiveSchedule,
    ExponentialSchedule,
    LinearSchedule,
    LogarithmicSchedule,
    SimulatedAnnealingPattern,
    TemperatureSchedule,
)

_SCHEDULES = [
    LinearSchedule(alpha=10.0),
    ExponentialSchedule(alpha=0.95),
    LogarithmicSchedule(c=100.0, d=2.0),
    AdaptiveSchedule(delta=0.5),
]
assert all(isinstance(s, TemperatureSchedule) for s in _SCHEDULES)


def _state_float(state: Mapping[str, object], key: str) -> float:
    value = state[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric.")
    return float(value)


def main() -> None:
    """Run one deterministic local simulated annealing workflow."""

    def objective_delegate(state: Mapping[str, object]) -> float:
        x = _state_float(state, "x")
        return (x - 3.0) ** 2

    def neighbor_delegate(state: Mapping[str, object]) -> Mapping[str, object]:
        x = _state_float(state, "x")
        step = 1.0 if x < 3.0 else -1.0
        return {"x": x + step}

    pattern = SimulatedAnnealingPattern(
        neighbor_delegate=neighbor_delegate,
        objective_delegate=objective_delegate,
        initial_state={"x": 0.0},
        expected_keys={"x"},
        state_validator=lambda state: -10.0 <= _state_float(state, "x") <= 10.0,
        initial_temperature=1.0,
        max_iterations=6,
        convergence_steps=3,
        random_seed=7,
    )
    result = pattern.run(
        "Minimize the distance from x to the target value 3.",
        request_id="example-pattern-simulated-annealing-001",
    )

    final_output = result.output["final_output"]
    print(
        json.dumps(
            {
                "success": result.success,
                "best_state": final_output["best_state"],
                "best_objective_value": final_output["best_objective_value"],
                "iterations": final_output["iterations"],
                "terminated_reason": result.output["terminated_reason"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
