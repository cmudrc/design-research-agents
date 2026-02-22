"""Example script.

Motivation
Run traced ``LoopStep`` composition for iterative design-threshold checks.

Diagram
```mermaid
flowchart LR
    A["Workflow input"] --> B["Workflow steps"]
    B --> C["workflow runtime loop step final output"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `workflow` use-cases and run `workflow_runtime_loop_step`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/workflow/workflow_runtime_loop_step.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from design_research_agents import LogicStep, LoopStep, Tracer, Workflow


def _increment_handler(context: Mapping[str, object]) -> Mapping[str, object]:
    loop_state = context.get("loop_state")
    state_mapping = loop_state if isinstance(loop_state, Mapping) else {}
    return {"counter": int(state_mapping.get("counter", 0)) + 1}


def _snapshot_handler(context: Mapping[str, object]) -> Mapping[str, object]:
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return {"counter": 0, "status": "looping"}
    increment_result = dependency_results.get("increment")
    if not isinstance(increment_result, Mapping):
        return {"counter": 0, "status": "looping"}
    increment_output = increment_result.get("output")
    if not isinstance(increment_output, Mapping):
        return {"counter": 0, "status": "looping"}
    counter = int(increment_output.get("counter", 0))
    return {
        "counter": counter,
        "status": "threshold_met" if counter >= 3 else "looping",
    }


def _state_reducer(
    state: Mapping[str, object],
    iteration_result: object,
    iteration: int,
) -> Mapping[str, object]:
    del state, iteration
    step_results = getattr(iteration_result, "step_results", {})
    if not isinstance(step_results, Mapping):
        return {"counter": 0}
    increment = step_results.get("increment")
    increment_output = getattr(increment, "output", {})
    if not isinstance(increment_output, Mapping):
        return {"counter": 0}
    return {
        "counter": int(increment_output.get("counter", 0)),
    }


def main() -> None:
    """Run a small loop and print compact JSON summary."""
    request_id = "example-workflow-loop-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    workflow = Workflow(
        tool_runtime=None,
        input_schema={"type": "object"},
        tracer=tracer,
        steps=[
            LoopStep(
                step_id="design_counter_loop",
                steps=(
                    LogicStep(step_id="increment", handler=_increment_handler),
                    LogicStep(
                        step_id="snapshot",
                        dependencies=("increment",),
                        handler=_snapshot_handler,
                    ),
                ),
                max_iterations=10,
                initial_state={"counter": 0},
                continue_predicate=lambda iteration, state: int(state.get("counter", 0)) < 3,
                state_reducer=_state_reducer,
                execution_mode="sequential",
                failure_policy="skip_dependents",
            )
        ],
    )

    result = workflow.run({}, execution_mode="sequential", request_id=request_id)
    loop_step = result.step_results.get("design_counter_loop")
    payload = {
        "example": "workflow/workflow_runtime_loop_step.py",
        "success": result.success,
        "execution_order": list(result.execution_order),
        "loop_status": loop_step.terminated_reason if loop_step else None,
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
