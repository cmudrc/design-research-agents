"""Sequential and DAG scheduling helpers for workflow runtime."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence

from design_research_agents._contracts._workflow import (
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowStep,
    WorkflowStepResult,
)

from ._workflow_graph import PreparedWorkflow, normalize_step_id, release_dependents

EvaluateStep = Callable[
    [
        WorkflowStep,
        str,
        Sequence[str],
        Mapping[str, WorkflowStepResult],
        set[str],
        PreparedWorkflow,
        Mapping[str, object],
        str,
        WorkflowExecutionMode,
        WorkflowFailurePolicy,
        Mapping[str, object],
    ],
    WorkflowStepResult,
]


def run_sequential(
    *,
    prepared: PreparedWorkflow,
    original_steps: Sequence[WorkflowStep],
    base_context: Mapping[str, object],
    resolved_request_id: str,
    resolved_dependencies: Mapping[str, object],
    failure_policy: WorkflowFailurePolicy,
    execution_mode: WorkflowExecutionMode,
    evaluate_step: EvaluateStep,
) -> tuple[dict[str, WorkflowStepResult], list[str]]:
    """Execute steps strictly in user-provided order."""
    step_results: dict[str, WorkflowStepResult] = {}
    execution_order: list[str] = []
    deactivated_steps: set[str] = set()

    for step in original_steps:
        step_id = normalize_step_id(step.step_id)
        step_dependencies = prepared.dependencies[step_id]

        unresolved_dependencies = [dependency for dependency in step_dependencies if dependency not in step_results]
        if unresolved_dependencies:
            raise ValueError(
                f"Step '{step_id}' cannot run before dependencies are resolved: "
                f"{', '.join(sorted(unresolved_dependencies))}."
            )

        # Sequential mode preserves author intent exactly; no reordering is attempted.
        step_result = evaluate_step(
            step,
            step_id,
            step_dependencies,
            step_results,
            deactivated_steps,
            prepared,
            base_context,
            resolved_request_id,
            execution_mode,
            failure_policy,
            resolved_dependencies,
        )
        step_results[step_id] = step_result
        execution_order.append(step_id)

    return step_results, execution_order


def run_dag(
    *,
    prepared: PreparedWorkflow,
    base_context: Mapping[str, object],
    resolved_request_id: str,
    resolved_dependencies: Mapping[str, object],
    failure_policy: WorkflowFailurePolicy,
    execution_mode: WorkflowExecutionMode,
    evaluate_step: EvaluateStep,
) -> tuple[dict[str, WorkflowStepResult], list[str]]:
    """Execute a DAG workflow using dependency-driven scheduling."""
    in_degree: dict[str, int] = {step_id: len(prepared.dependencies[step_id]) for step_id in prepared.step_map}
    ready_steps = [step_id for step_id, degree in in_degree.items() if degree == 0]
    # Heap-based frontier keeps tie-breaking deterministic across Python versions/platforms.
    heapq.heapify(ready_steps)

    step_results: dict[str, WorkflowStepResult] = {}
    execution_order: list[str] = []
    deactivated_steps: set[str] = set()

    while ready_steps:
        step_id = heapq.heappop(ready_steps)
        if step_id in step_results:
            continue

        step = prepared.step_map[step_id]
        step_dependencies = prepared.dependencies[step_id]

        # Evaluate only when all predecessors have released this node.
        step_result = evaluate_step(
            step,
            step_id,
            step_dependencies,
            step_results,
            deactivated_steps,
            prepared,
            base_context,
            resolved_request_id,
            execution_mode,
            failure_policy,
            resolved_dependencies,
        )

        step_results[step_id] = step_result
        execution_order.append(step_id)
        release_dependents(
            step_id=step_id,
            dependents=prepared.dependents,
            in_degree=in_degree,
            ready_steps=ready_steps,
        )

    if len(step_results) != len(prepared.step_map):
        unresolved_steps = sorted(set(prepared.step_map).difference(step_results))
        raise RuntimeError(f"DAG workflow execution ended with unresolved steps: {', '.join(unresolved_steps)}")

    return step_results, execution_order


__all__ = ["run_dag", "run_sequential"]
