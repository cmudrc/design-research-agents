"""Runnable entrypoint demonstrating ``LoopStep`` composition."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.contracts import ExecutionResult, LogicStep, LoopStep
from design_research_agents.workflow import Workflow


def main() -> None:
    """Run a small loop that increments a counter until a stop condition is met."""
    workflow = Workflow(
        tool_runtime=None,
        input_mode="schema",
        steps=[
            LoopStep(
                step_id="counter_loop",
                steps=(
                    LogicStep(
                        step_id="increment",
                        handler=_increment_handler,
                    ),
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

    workflow_result = workflow.run(
        {},
        execution_mode="sequential",
    )
    print(workflow_result.asdict())


def _increment_handler(context: Mapping[str, object]) -> Mapping[str, object]:
    """Run increment handler.

    Args:
        context: Parameter value.

    Returns:
        The resulting value.
    """
    loop_state = context.get("loop_state")
    state_mapping = loop_state if isinstance(loop_state, Mapping) else {}
    return {"counter": int(state_mapping.get("counter", 0)) + 1}


def _snapshot_handler(context: Mapping[str, object]) -> Mapping[str, object]:
    """Run snapshot handler.

    Args:
        context: Parameter value.

    Returns:
        The resulting value.
    """
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return {"counter": 0, "status": "looping"}
    increment_result = dependency_results.get("increment")
    if not isinstance(increment_result, Mapping):
        return {"counter": 0, "status": "looping"}
    increment_output = increment_result.get("output")
    if not isinstance(increment_output, Mapping):
        return {"counter": 0, "status": "looping"}
    return {
        "counter": int(increment_output.get("counter", 0)),
        "status": "looping",
    }


def _state_reducer(
    state: Mapping[str, object],
    iteration_result: ExecutionResult,
    iteration: int,
) -> Mapping[str, object]:
    """Carry the latest counter value into the next iteration.

    Args:
        state: Parameter value.
        iteration_result: Parameter value.
        iteration: Parameter value.

    Returns:
        The resulting value.
    """
    del state, iteration
    return {
        "counter": int(iteration_result.step_results["increment"].output["counter"]),
    }


if __name__ == "__main__":
    main()
