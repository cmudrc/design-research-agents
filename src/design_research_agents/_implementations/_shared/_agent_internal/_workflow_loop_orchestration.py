"""Workflow-loop orchestration helpers for multi-step agent implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._workflow import LogicStep, LoopStep
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import Workflow

LoopContinuePredicate = Callable[[int, Mapping[str, object]], bool]
LoopIterationHandler = Callable[[int, Mapping[str, object]], Mapping[str, object]]


@dataclass(slots=True, frozen=True, kw_only=True)
class WorkflowLoopResult:
    """Result returned by ``run_workflow_loop``."""

    final_state: dict[str, object]
    """Final loop state after iteration termination."""
    terminated_reason: str
    """Loop termination reason emitted by ``LoopStep``."""
    iterations_executed: int
    """Number of iterations executed by the loop."""
    workflow_result: ExecutionResult
    """Underlying workflow execution result for the loop wrapper."""
    workflow: Workflow
    """Composed public workflow object used for the loop run."""


def run_workflow_loop(
    *,
    max_iterations: int,
    initial_state: Mapping[str, object],
    continue_predicate: LoopContinuePredicate,
    iteration_handler: LoopIterationHandler,
    request_id: str,
    dependencies: Mapping[str, object],
    tracer: Tracer | None = None,
) -> WorkflowLoopResult:
    """A stateful iteration loop through public ``Workflow`` + ``LoopStep``.

    Args:
        max_iterations: Maximum number of loop iterations.
        initial_state: Initial loop state mapping.
        continue_predicate: Predicate controlling whether each iteration runs.
        iteration_handler: Handler producing next loop state per iteration.
        request_id: Request identifier used for nested runtime correlation.
        dependencies: Dependency payload passed to nested executions.
        tracer: Optional tracer dependency.

    Returns:
        Normalized loop orchestration result with final state and termination info.
    """
    workflow = compile_workflow_loop(
        max_iterations=max_iterations,
        initial_state=initial_state,
        continue_predicate=continue_predicate,
        iteration_handler=iteration_handler,
        tracer=tracer,
    )

    workflow_result = workflow.run(
        {},
        execution_mode="sequential",
        failure_policy="skip_dependents",
        request_id=f"{request_id}:workflow_loop",
        dependencies=dependencies,
    )

    loop_step_result = workflow_result.step_results.get("agent_loop")
    loop_output = loop_step_result.output if loop_step_result is not None else {}
    final_state_raw = loop_output.get("final_state", {})
    final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}
    terminated_reason = str(loop_output.get("terminated_reason", "max_iterations_reached"))
    iterations_executed = int(loop_output.get("iterations_executed", 0))
    return WorkflowLoopResult(
        final_state=final_state,
        terminated_reason=terminated_reason,
        iterations_executed=iterations_executed,
        workflow_result=workflow_result,
        workflow=workflow,
    )


def compile_workflow_loop(
    *,
    max_iterations: int,
    initial_state: Mapping[str, object],
    continue_predicate: LoopContinuePredicate,
    iteration_handler: LoopIterationHandler,
    tracer: Tracer | None = None,
) -> Workflow:
    """Compile a stateful iteration loop into a public ``Workflow``."""

    def _iteration_logic(step_context: Mapping[str, object]) -> Mapping[str, object]:
        """One loop iteration handler with normalized loop state."""
        loop_metadata = step_context.get("_loop")
        raw_iteration = loop_metadata.get("iteration") if isinstance(loop_metadata, Mapping) else 1
        iteration = _parse_iteration(raw_iteration)
        loop_state = step_context.get("loop_state")
        current_state = dict(loop_state) if isinstance(loop_state, Mapping) else {}
        next_state = iteration_handler(iteration, current_state)
        if not isinstance(next_state, Mapping):
            raise ValueError("Loop iteration handler must return a mapping state.")
        return dict(next_state)

    def _state_reducer(
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        """Reduce loop state after one iteration workflow result."""
        del iteration
        iteration_step = iteration_result.step_results.get("loop_iteration")
        if iteration_step is None or not getattr(iteration_step, "success", False):
            return dict(state)
        output = getattr(iteration_step, "output", {})
        return dict(output) if isinstance(output, Mapping) else dict(state)

    return Workflow(
        tool_runtime=None,
        tracer=tracer,
        input_schema={"type": "object"},
        steps=[
            LoopStep(
                step_id="agent_loop",
                steps=(
                    LogicStep(
                        step_id="loop_iteration",
                        handler=_iteration_logic,
                    ),
                ),
                max_iterations=max_iterations,
                initial_state=dict(initial_state),
                continue_predicate=continue_predicate,
                state_reducer=_state_reducer,
                execution_mode="sequential",
                failure_policy="skip_dependents",
            )
        ],
    )


def _parse_iteration(raw_iteration: object) -> int:
    """Parse raw loop iteration metadata into a positive integer.

    Args:
        raw_iteration: Raw iteration value emitted by runtime metadata.

    Returns:
        Parsed positive iteration number, defaulting to ``1``.
    """
    if isinstance(raw_iteration, int) and raw_iteration > 0:
        return raw_iteration
    if isinstance(raw_iteration, str) and raw_iteration.strip().isdigit():
        parsed = int(raw_iteration.strip())
        if parsed > 0:
            return parsed
    return 1


__all__ = [
    "LoopContinuePredicate",
    "LoopIterationHandler",
    "WorkflowLoopResult",
    "compile_workflow_loop",
    "run_workflow_loop",
]
