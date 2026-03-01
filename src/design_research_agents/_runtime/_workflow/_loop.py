"""Loop-step execution helper for workflow runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from design_research_agents._contracts._workflow import (
    ExecutionResult,
    LoopStep,
    WorkflowStepResult,
)

RunWorkflow = Callable[..., ExecutionResult]


def run_loop_step(
    *,
    run_workflow: RunWorkflow,
    step: LoopStep,
    step_id: str,
    step_context: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute a loop step by repeatedly running its nested workflow body."""
    if step.max_iterations < 1:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
            error="LoopStep max_iterations must be >= 1.",
            metadata={"stage": "loop_binding"},
        )

    if not step.steps:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
            error="LoopStep requires at least one nested step.",
            metadata={"stage": "loop_binding"},
        )

    current_state = dict(step.initial_state or {})
    iteration_results: list[ExecutionResult] = []
    terminated_reason = "max_iterations_reached"
    parent_dependency_results = step_context.get("dependency_results")
    parent_dependency_snapshot = (
        dict(parent_dependency_results) if isinstance(parent_dependency_results, Mapping) else {}
    )

    for iteration in range(1, step.max_iterations + 1):
        # Evaluate predicate at iteration start so reducers never run for skipped iterations.
        if step.continue_predicate is not None and not step.continue_predicate(
            iteration,
            dict(current_state),
        ):
            terminated_reason = "condition_stopped"
            break

        loop_context = dict(step_context)
        # Inject loop metadata into context to let nested steps introspect iteration state deterministically.
        loop_context["loop_state"] = dict(current_state)
        loop_context["_loop"] = {
            "loop_step_id": step_id,
            "iteration": iteration,
            "max_iterations": step.max_iterations,
            "execution_mode": step.execution_mode,
            "failure_policy": step.failure_policy,
        }
        loop_context["loop_parent_dependency_results"] = dict(parent_dependency_snapshot)

        iteration_result = run_workflow(
            step.steps,
            context=loop_context,
            execution_mode=step.execution_mode,
            failure_policy=step.failure_policy,
            request_id=f"{request_id}:workflow:{step_id}:loop:{iteration}",
            dependencies=dependencies,
        )
        iteration_results.append(iteration_result)

        if step.state_reducer is not None:
            # Reducer output becomes the full next-state payload (not a patch merge).
            reduced_state = step.state_reducer(
                dict(current_state),
                iteration_result,
                iteration,
            )
            if not isinstance(reduced_state, Mapping):
                return WorkflowStepResult(
                    step_id=step_id,
                    status="failed",
                    success=False,
                    output={},
                    error="LoopStep state_reducer must return a mapping.",
                    metadata={"stage": "loop_state_reducer"},
                )
            current_state = dict(reduced_state)

        if not iteration_result.success:
            terminated_reason = "iteration_failed"
            break

    output = {
        "success": terminated_reason != "iteration_failed",
        "iterations": step.max_iterations,
        "iterations_executed": len(iteration_results),
        "terminated_reason": terminated_reason,
        "final_state": dict(current_state),
        "iteration_results": [result.to_dict() for result in iteration_results],
    }
    if terminated_reason == "iteration_failed":
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output=output,
            error="Loop iteration failed.",
            metadata={"stage": "loop_execution"},
        )
    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=output,
        metadata={"stage": "loop_execution"},
    )


__all__ = ["run_loop_step"]
