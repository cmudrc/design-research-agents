"""Deterministic workflow runtime for tool, agent, and logic steps."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence

from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    resolve_request_id,
)
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import (
    AgentStep,
    LogicStep,
    LoopStep,
    WorkflowDelegate,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowResult,
    WorkflowRunner,
    WorkflowStep,
    WorkflowStepResult,
)
from design_research_agents.tracing import Tracer, finish_trace_run, start_trace_run
from design_research_agents.workflow.internal import (
    PreparedWorkflow,
    activate_step_span,
    build_step_context,
    finish_step_span,
    has_upstream_failure,
    normalize_step_id,
    prepare_workflow_graph,
    release_dependents,
    route_deactivations,
    run_agent_step,
    run_logic_step,
    run_tool_step,
    start_step_span,
    validate_no_cycles,
)


class WorkflowRuntime(WorkflowRunner):
    """Execute typed workflows with deterministic sequential or DAG scheduling."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime | None = None,
        agents: Mapping[str, WorkflowDelegate] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize workflow runtime dependencies for tool and delegate steps."""
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self._agents = {
            name.strip(): agent
            for name, agent in (agents or {}).items()
            if isinstance(name, str) and name.strip()
        }

    def run(
        self,
        steps: Sequence[WorkflowStep],
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "dag",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute one workflow definition and return aggregated results."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        base_context = dict(context or {})
        trace_scope = start_trace_run(
            agent_name="WorkflowRuntime",
            request_id=resolved_request_id,
            input_payload={
                "step_count": len(steps),
                "execution_mode": execution_mode,
                "failure_policy": failure_policy,
            },
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )

        try:
            prepared = prepare_workflow_graph(steps)
            if execution_mode == "dag":
                validate_no_cycles(prepared.step_map, prepared.dependencies)

            if execution_mode == "sequential":
                step_results, execution_order = self._run_sequential(
                    prepared=prepared,
                    original_steps=steps,
                    base_context=base_context,
                    resolved_request_id=resolved_request_id,
                    resolved_dependencies=resolved_dependencies,
                    failure_policy=failure_policy,
                    execution_mode=execution_mode,
                )
            elif execution_mode == "dag":
                step_results, execution_order = self._run_dag(
                    prepared=prepared,
                    base_context=base_context,
                    resolved_request_id=resolved_request_id,
                    resolved_dependencies=resolved_dependencies,
                    failure_policy=failure_policy,
                    execution_mode=execution_mode,
                )
            else:
                raise ValueError(f"Unsupported execution_mode '{execution_mode}'.")
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        success = all(
            result.success or result.error == "skipped_branch_not_selected"
            for result in step_results.values()
        )
        workflow_result = WorkflowResult(
            success=success,
            step_results=step_results,
            execution_order=execution_order,
            metadata={
                "runtime": "workflow",
                "execution_mode": execution_mode,
                "failure_policy": failure_policy,
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "step_count": len(steps),
            },
        )
        finish_trace_run(trace_scope, result=workflow_result)
        return workflow_result

    def _run_sequential(
        self,
        *,
        prepared: PreparedWorkflow,
        original_steps: Sequence[WorkflowStep],
        base_context: Mapping[str, object],
        resolved_request_id: str,
        resolved_dependencies: Mapping[str, object],
        failure_policy: WorkflowFailurePolicy,
        execution_mode: WorkflowExecutionMode,
    ) -> tuple[dict[str, WorkflowStepResult], list[str]]:
        step_results: dict[str, WorkflowStepResult] = {}
        execution_order: list[str] = []
        deactivated_steps: set[str] = set()

        for step in original_steps:
            step_id = normalize_step_id(step.step_id)
            step_dependencies = prepared.dependencies[step_id]

            unresolved_dependencies = [
                dependency for dependency in step_dependencies if dependency not in step_results
            ]
            if unresolved_dependencies:
                raise ValueError(
                    f"Step '{step_id}' cannot run before dependencies are resolved: "
                    f"{', '.join(sorted(unresolved_dependencies))}."
                )

            step_result = self._evaluate_step(
                step=step,
                step_id=step_id,
                step_dependencies=step_dependencies,
                step_results=step_results,
                deactivated_steps=deactivated_steps,
                prepared=prepared,
                base_context=base_context,
                request_id=resolved_request_id,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
                dependencies=resolved_dependencies,
            )

            step_results[step_id] = step_result
            execution_order.append(step_id)

        return step_results, execution_order

    def _run_dag(
        self,
        *,
        prepared: PreparedWorkflow,
        base_context: Mapping[str, object],
        resolved_request_id: str,
        resolved_dependencies: Mapping[str, object],
        failure_policy: WorkflowFailurePolicy,
        execution_mode: WorkflowExecutionMode,
    ) -> tuple[dict[str, WorkflowStepResult], list[str]]:
        in_degree: dict[str, int] = {
            step_id: len(prepared.dependencies[step_id]) for step_id in prepared.step_map
        }
        ready_steps = [step_id for step_id, degree in in_degree.items() if degree == 0]
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

            step_result = self._evaluate_step(
                step=step,
                step_id=step_id,
                step_dependencies=step_dependencies,
                step_results=step_results,
                deactivated_steps=deactivated_steps,
                prepared=prepared,
                base_context=base_context,
                request_id=resolved_request_id,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
                dependencies=resolved_dependencies,
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
            raise RuntimeError(
                f"DAG workflow execution ended with unresolved steps: {', '.join(unresolved_steps)}"
            )

        return step_results, execution_order

    def _execute_step(
        self,
        *,
        step: WorkflowStep,
        step_id: str,
        step_context: Mapping[str, object],
        request_id: str,
        execution_mode: WorkflowExecutionMode,
        failure_policy: WorkflowFailurePolicy,
        dependencies: Mapping[str, object],
    ) -> WorkflowStepResult:
        step_span_id = start_step_span(step=step, step_id=step_id)
        with activate_step_span(step_span_id):
            if isinstance(step, LogicStep):
                result = run_logic_step(step=step, step_id=step_id, step_context=step_context)
            elif isinstance(step, AgentStep):
                result = run_agent_step(
                    agents=self._agents,
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                    request_id=request_id,
                    execution_mode=execution_mode,
                    failure_policy=failure_policy,
                    dependencies=dependencies,
                )
            elif isinstance(step, LoopStep):
                result = self._run_loop_step(
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                    request_id=request_id,
                    dependencies=dependencies,
                )
            else:
                result = run_tool_step(
                    tool_runtime=self._tool_runtime,
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                    request_id=request_id,
                    execution_mode=execution_mode,
                    failure_policy=failure_policy,
                    dependencies=dependencies,
                )

        finish_step_span(
            span_id=step_span_id,
            step_id=step_id,
            status=result.status,
            error=result.error,
        )
        return result

    def _run_loop_step(
        self,
        *,
        step: LoopStep,
        step_id: str,
        step_context: Mapping[str, object],
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> WorkflowStepResult:
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
        iteration_results: list[WorkflowResult] = []
        terminated_reason = "max_iterations_reached"
        parent_dependency_results = step_context.get("dependency_results")
        parent_dependency_snapshot = (
            dict(parent_dependency_results)
            if isinstance(parent_dependency_results, Mapping)
            else {}
        )

        for iteration in range(1, step.max_iterations + 1):
            if step.continue_predicate is not None and not step.continue_predicate(
                iteration,
                dict(current_state),
            ):
                terminated_reason = "condition_stopped"
                break

            loop_context = dict(step_context)
            loop_context["loop_state"] = dict(current_state)
            loop_context["_loop"] = {
                "loop_step_id": step_id,
                "iteration": iteration,
                "max_iterations": step.max_iterations,
                "execution_mode": step.execution_mode,
                "failure_policy": step.failure_policy,
            }
            loop_context["loop_parent_dependency_results"] = dict(parent_dependency_snapshot)

            iteration_result = self.run(
                step.steps,
                context=loop_context,
                execution_mode=step.execution_mode,
                failure_policy=step.failure_policy,
                request_id=f"{request_id}:workflow:{step_id}:loop:{iteration}",
                dependencies=dependencies,
            )
            iteration_results.append(iteration_result)

            if step.state_reducer is not None:
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
            "iteration_results": [result.asdict() for result in iteration_results],
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

    def _evaluate_step(
        self,
        *,
        step: WorkflowStep,
        step_id: str,
        step_dependencies: Sequence[str],
        step_results: Mapping[str, WorkflowStepResult],
        deactivated_steps: set[str],
        prepared: PreparedWorkflow,
        base_context: Mapping[str, object],
        request_id: str,
        execution_mode: WorkflowExecutionMode,
        failure_policy: WorkflowFailurePolicy,
        dependencies: Mapping[str, object],
    ) -> WorkflowStepResult:
        if step_id in deactivated_steps:
            return self._skip_step_result(step_id=step_id, reason="skipped_branch_not_selected")

        if failure_policy == "skip_dependents" and has_upstream_failure(
            dependencies=step_dependencies,
            step_results=step_results,
        ):
            return self._skip_step_result(step_id=step_id, reason="skipped_upstream_failure")

        step_context = build_step_context(
            base_context=base_context,
            step_id=step_id,
            step_dependencies=step_dependencies,
            step_results=step_results,
            request_id=request_id,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
        )
        step_result = self._execute_step(
            step=step,
            step_id=step_id,
            step_context=step_context,
            request_id=request_id,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            dependencies=dependencies,
        )

        if not step_result.success or not isinstance(step, LogicStep):
            return step_result

        deactivation_update, route_error = route_deactivations(
            step=step,
            step_output=step_result.output,
            dependents=prepared.dependents,
        )
        if route_error is not None:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output=dict(step_result.output),
                error=route_error,
                metadata={"stage": "routing"},
            )
        deactivated_steps.update(deactivation_update)
        return step_result

    def _skip_step_result(self, *, step_id: str, reason: str) -> WorkflowStepResult:
        return WorkflowStepResult(
            step_id=step_id,
            status="skipped",
            success=False,
            output={},
            error=reason,
        )
