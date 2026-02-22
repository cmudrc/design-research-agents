r"""# Workflow / Workflow Runtime Loop Step.

## Introduction
Tree of Thoughts and ReAct each motivate iterative reasoning with explicit state updates, and AutoGen
provides a practical framing for orchestrating repeated loop actions. This example demonstrates loop-step
execution in the workflow runtime, including bounded iteration behavior and trace emission.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["Workflow.run(...)"]
    C --> D["WorkflowRuntime schedules step graph (LogicStep, LoopStep)"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "workflow/workflow_runtime_loop_step.py",
     "execution_order": [
       "design_counter_loop"
     ],
     "final_output": {
       "final_state": {
         "counter": 3
       },
       "iteration_results": [
         {
           "execution_order": [
             "increment",
             "snapshot"
           ],
           "metadata": {
             "artifact_count": 0,
             "dependency_keys": [],
             "execution_mode": "sequential",
             "failure_policy": "skip_dependents",
             "memory_enabled": false,
             "request_id": "example-workflow-loop-design-001:workflow:design_counter_loop:loop:1",
             "runtime": "workflow",
             "step_count": 2
           },
           "model_response": null,
           "output": {
             "artifacts": [],
             "final_output": {
               "counter": 1,
               "status": "looping"
             },
             "workflow": {
               "execution_order": [
                 "increment",
                 "snapshot"
               ],
               "step_results": {
                 "increment": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 1
                   },
                   "status": "completed",
                   "step_id": "increment",
                   "success": true
                 },
                 "snapshot": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 1,
                     "status": "looping"
                   },
                   "status": "completed",
                   "step_id": "snapshot",
                   "success": true
                 }
               },
               "success": true
             }
           },
           "step_results": {
             "increment": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 1
               },
               "status": "completed",
               "step_id": "increment",
               "success": true
             },
             "snapshot": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 1,
                 "status": "looping"
               },
               "status": "completed",
               "step_id": "snapshot",
               "success": true
             }
           },
           "success": true,
           "tool_results": []
         },
         {
           "execution_order": [
             "increment",
             "snapshot"
           ],
           "metadata": {
             "artifact_count": 0,
             "dependency_keys": [],
             "execution_mode": "sequential",
             "failure_policy": "skip_dependents",
             "memory_enabled": false,
             "request_id": "example-workflow-loop-design-001:workflow:design_counter_loop:loop:2",
             "runtime": "workflow",
             "step_count": 2
           },
           "model_response": null,
           "output": {
             "artifacts": [],
             "final_output": {
               "counter": 2,
               "status": "looping"
             },
             "workflow": {
               "execution_order": [
                 "increment",
                 "snapshot"
               ],
               "step_results": {
                 "increment": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 2
                   },
                   "status": "completed",
                   "step_id": "increment",
                   "success": true
                 },
                 "snapshot": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 2,
                     "status": "looping"
                   },
                   "status": "completed",
                   "step_id": "snapshot",
                   "success": true
                 }
               },
               "success": true
             }
           },
           "step_results": {
             "increment": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 2
               },
               "status": "completed",
               "step_id": "increment",
               "success": true
             },
             "snapshot": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 2,
                 "status": "looping"
               },
               "status": "completed",
               "step_id": "snapshot",
               "success": true
             }
           },
           "success": true,
           "tool_results": []
         },
         {
           "execution_order": [
             "increment",
             "snapshot"
           ],
           "metadata": {
             "artifact_count": 0,
             "dependency_keys": [],
             "execution_mode": "sequential",
             "failure_policy": "skip_dependents",
             "memory_enabled": false,
             "request_id": "example-workflow-loop-design-001:workflow:design_counter_loop:loop:3",
             "runtime": "workflow",
             "step_count": 2
           },
           "model_response": null,
           "output": {
             "artifacts": [],
             "final_output": {
               "counter": 3,
               "status": "threshold_met"
             },
             "workflow": {
               "execution_order": [
                 "increment",
                 "snapshot"
               ],
               "step_results": {
                 "increment": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 3
                   },
                   "status": "completed",
                   "step_id": "increment",
                   "success": true
                 },
                 "snapshot": {
                   "artifacts": [],
                   "error": null,
                   "metadata": {
                     "stage": "execution"
                   },
                   "output": {
                     "counter": 3,
                     "status": "threshold_met"
                   },
                   "status": "completed",
                   "step_id": "snapshot",
                   "success": true
                 }
               },
               "success": true
             }
           },
           "step_results": {
             "increment": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 3
               },
               "status": "completed",
               "step_id": "increment",
               "success": true
             },
             "snapshot": {
               "artifacts": [],
               "error": null,
               "metadata": {
                 "stage": "execution"
               },
               "output": {
                 "counter": 3,
                 "status": "threshold_met"
               },
               "status": "completed",
               "step_id": "snapshot",
               "success": true
             }
           },
           "success": true,
           "tool_results": []
         }
       ],
       "iterations": 10,
       "iterations_executed": 3,
       "success": true,
       "terminated_reason": "condition_stopped"
     },
     "loop_status": "condition_stopped",
     "success": true,
     "terminated_reason": null,
     "trace": {
       "request_id": "example-workflow-loop-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-loop-design-001.jsonl"
     }
   }


## References
- `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
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
