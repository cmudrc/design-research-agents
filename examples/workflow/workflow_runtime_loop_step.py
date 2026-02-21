"""Run traced ``LoopStep`` composition for iterative design-threshold checks.

Expected observations:
- ``counter`` reaches the loop stop condition.
- ``terminated_reason`` reflects loop termination semantics.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents import LogicStep, LoopStep, Workflow
from design_research_agents.contracts import ExecutionResult
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


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
    iteration_result: ExecutionResult,
    iteration: int,
) -> Mapping[str, object]:
    del state, iteration
    return {
        "counter": int(iteration_result.step_results["increment"].output["counter"]),
    }


def main() -> None:
    """Run a small loop and print compact JSON summary."""
    request_id = "example-workflow-loop-design-001"
    workflow = Workflow(
        tool_runtime=None,
        input_mode="schema",
        tracer=make_tracer(),
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
        "loop_status": loop_step.output.get("terminated_reason") if loop_step else None,
        "final_output": (
            result.output.get("final_output") if isinstance(result.output, dict) else None
        ),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
