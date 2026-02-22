"""Deterministic workflow runtime for tool, agent, logic, loop, and memory steps."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from design_research_agents._contracts._memory import MemoryStore
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._contracts._workflow import (
    AgentStep,
    DelegateBatchStep,
    ExecutionResult,
    LogicStep,
    LoopStep,
    MemoryReadStep,
    MemoryWriteStep,
    ModelStep,
    WorkflowArtifact,
    WorkflowArtifactSource,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowRunner,
    WorkflowStep,
    WorkflowStepResult,
)
from design_research_agents._tracing import (
    Tracer,
    emit_workflow_step_context,
    emit_workflow_step_result,
    finish_trace_run,
    start_trace_run,
)

from ._artifacts import collect_artifacts, dedupe_artifacts, resolve_final_output
from ._executors._common import (
    run_agent_step,
    run_delegate_batch_step,
    run_logic_step,
    run_memory_read_step,
    run_memory_write_step,
    run_model_step,
    run_tool_step,
)
from ._loop import run_loop_step
from ._runtime_options import normalize_dependencies, resolve_request_id
from ._step_context import build_step_context, has_upstream_failure, route_deactivations
from ._step_tracing import activate_step_span, finish_step_span, start_step_span, step_kind
from ._workflow_graph import (
    PreparedWorkflow,
    normalize_step_id,
    prepare_workflow_graph,
    release_dependents,
    validate_no_cycles,
)


class WorkflowRuntime(WorkflowRunner):
    """Execute typed workflows with deterministic sequential or DAG scheduling."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime | None = None,
        memory_store: MemoryStore | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize workflow runtime dependencies for tool and delegate steps.

        Args:
            tool_runtime: Optional tool runtime used for ``ToolStep`` execution.
            memory_store: Optional memory store used for memory step execution.
            tracer: Optional tracer used for workflow and step span emission.
        """
        self._tool_runtime = tool_runtime
        self._memory_store = memory_store
        self._tracer = tracer

    def run(
        self,
        steps: Sequence[WorkflowStep],
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "dag",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one workflow definition and return aggregated results.

        Args:
            steps: Workflow step definitions to execute.
            context: Optional shared context mapping available to all step builders.
            execution_mode: Scheduling mode (``sequential`` or ``dag``).
            failure_policy: Upstream-failure handling strategy for dependents.
            request_id: Optional external request id for trace correlation.
            dependencies: Optional dependency payloads exposed to step executions.

        Returns:
            Aggregated workflow result containing per-step outputs and metadata.

        Raises:
            ValueError: If an unsupported execution mode is requested.
            RuntimeError: If DAG execution leaves unresolved steps.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        base_context = dict(context or {})
        # Emit one top-level trace scope so all step spans share a common request lineage.
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

        # Branch-skipped steps are represented as non-success with a sentinel error; treat those as expected.
        success = all(
            result.success or result.error == "skipped_branch_not_selected" for result in step_results.values()
        )
        artifact_records = collect_artifacts(
            step_results=step_results,
            execution_order=execution_order,
        )
        output: dict[str, object] = {
            "workflow": {
                "success": success,
                "execution_order": list(execution_order),
                "step_results": {step_id: result.to_dict() for step_id, result in step_results.items()},
            },
            "final_output": resolve_final_output(
                step_results=step_results,
                execution_order=execution_order,
            ),
            "artifacts": [asdict(record) for record in artifact_records],
        }
        workflow_result = ExecutionResult(
            success=success,
            output=output,
            step_results=step_results,
            execution_order=execution_order,
            metadata={
                "runtime": "workflow",
                "execution_mode": execution_mode,
                "failure_policy": failure_policy,
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "step_count": len(steps),
                "memory_enabled": self._memory_store is not None,
                "artifact_count": len(artifact_records),
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
        """Execute steps strictly in the user-provided order.

        Args:
            prepared: Precomputed workflow graph metadata.
            original_steps: Original ordered step sequence from caller input.
            base_context: Shared workflow context mapping.
            resolved_request_id: Resolved request id used for nested calls.
            resolved_dependencies: Resolved dependency payload mapping.
            failure_policy: Upstream-failure handling strategy.
            execution_mode: Scheduling mode used for downstream step execution.

        Returns:
            Tuple of ``(step_results, execution_order)``.

        Raises:
            ValueError: If a step is encountered before one of its dependencies.
        """
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
        """Execute a DAG workflow using dependency-driven scheduling.

        Args:
            prepared: Precomputed workflow graph metadata.
            base_context: Shared workflow context mapping.
            resolved_request_id: Resolved request id used for nested calls.
            resolved_dependencies: Resolved dependency payload mapping.
            failure_policy: Upstream-failure handling strategy.
            execution_mode: Scheduling mode used for downstream step execution.

        Returns:
            Tuple of ``(step_results, execution_order)``.

        Raises:
            RuntimeError: If execution terminates before all DAG nodes resolve.
        """
        in_degree: dict[str, int] = {step_id: len(prepared.dependencies[step_id]) for step_id in prepared.step_map}
        ready_steps = [step_id for step_id, degree in in_degree.items() if degree == 0]
        # Heap ordering keeps DAG execution deterministic when multiple nodes become ready together.
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
            raise RuntimeError(f"DAG workflow execution ended with unresolved steps: {', '.join(unresolved_steps)}")

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
        """Dispatch one normalized step to its concrete execution handler.

        Args:
            step: Normalized workflow step definition.
            step_id: Canonical step id for trace/result bookkeeping.
            step_context: Runtime step context built from dependencies and base context.
            request_id: Workflow request id used for downstream calls.
            execution_mode: Scheduling mode propagated to nested executions.
            failure_policy: Upstream-failure policy propagated to nested executions.
            dependencies: External dependency payload mapping available to delegates.

        Returns:
            Step execution result payload.
        """
        step_span_id = start_step_span(step=step, step_id=step_id)
        with activate_step_span(step_span_id):
            resolved_step_type = step_kind(step)
            emit_workflow_step_context(
                step_id=step_id,
                step_type=resolved_step_type,
                context=step_context,
            )
            if isinstance(step, LogicStep):
                result = run_logic_step(step=step, step_id=step_id, step_context=step_context)
            elif isinstance(step, AgentStep):
                result = run_agent_step(
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                    request_id=request_id,
                    execution_mode=execution_mode,
                    failure_policy=failure_policy,
                    dependencies=dependencies,
                )
            elif isinstance(step, ModelStep):
                result = run_model_step(step=step, step_id=step_id, step_context=step_context)
            elif isinstance(step, DelegateBatchStep):
                result = run_delegate_batch_step(
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                    request_id=request_id,
                    execution_mode=execution_mode,
                    failure_policy=failure_policy,
                    dependencies=dependencies,
                )
            elif isinstance(step, MemoryReadStep):
                result = run_memory_read_step(
                    memory_store=self._memory_store,
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                )
            elif isinstance(step, MemoryWriteStep):
                result = run_memory_write_step(
                    memory_store=self._memory_store,
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                )
            elif isinstance(step, LoopStep):
                result = run_loop_step(
                    run_workflow=self.run,
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
            emit_workflow_step_result(
                step_id=step_id,
                step_type=resolved_step_type,
                status=result.status,
                success=result.success,
                output=result.output,
                error=result.error,
                metadata=result.metadata,
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
        """Execute a ``LoopStep`` by repeatedly running its nested body workflow.

        Args:
            step: Loop step definition with body, state hooks, and limits.
            step_id: Canonical step id for trace/result bookkeeping.
            step_context: Runtime step context from outer workflow.
            request_id: Workflow request id prefix for nested loop iterations.
            dependencies: External dependency payload mapping for nested runs.

        Returns:
            Loop step result including termination metadata and iteration summaries.
        """
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
        """Evaluate preconditions, build context, and execute one step.

        Args:
            step: Normalized workflow step definition.
            step_id: Canonical step id for trace/result bookkeeping.
            step_dependencies: Dependency step ids required before execution.
            step_results: Results accumulated so far.
            deactivated_steps: Steps deactivated by prior routing decisions.
            prepared: Precomputed workflow graph metadata.
            base_context: Shared workflow context mapping.
            request_id: Workflow request id used for downstream calls.
            execution_mode: Scheduling mode propagated to nested executions.
            failure_policy: Upstream-failure handling strategy.
            dependencies: External dependency payload mapping for delegates.

        Returns:
            Step result, including skipped/failed outcomes when preconditions fail.
        """
        if step_id in deactivated_steps:
            return self._skip_step_result(step_id=step_id, reason="skipped_branch_not_selected")

        if failure_policy == "skip_dependents" and has_upstream_failure(
            dependencies=step_dependencies,
            step_results=step_results,
        ):
            return self._skip_step_result(step_id=step_id, reason="skipped_upstream_failure")

        raw_output_schema = base_context.get("_workflow_output_schema")
        resolved_output_schema = raw_output_schema if isinstance(raw_output_schema, Mapping) else None
        step_context = build_step_context(
            base_context=base_context,
            step_id=step_id,
            step_dependencies=step_dependencies,
            step_results=step_results,
            request_id=request_id,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            is_terminal_step=not prepared.dependents.get(step_id),
            output_schema=resolved_output_schema,
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
        step_result = self._apply_step_artifacts(
            step=step,
            step_result=step_result,
            step_context=step_context,
        )

        if not step_result.success or not isinstance(step, LogicStep):
            return step_result

        # Only logic steps participate in route-map branch selection.
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
        """Construct a standardized skipped-step result.

        Args:
            step_id: Canonical step id for the skipped step.
            reason: Machine-readable skip reason.

        Returns:
            Skipped ``WorkflowStepResult`` payload.
        """
        return WorkflowStepResult(
            step_id=step_id,
            status="skipped",
            success=False,
            output={},
            error=reason,
        )

    def _apply_step_artifacts(
        self,
        *,
        step: WorkflowStep,
        step_result: WorkflowStepResult,
        step_context: Mapping[str, object],
    ) -> WorkflowStepResult:
        """Attach normalized artifacts to a step result.

        Args:
            step: Step definition used for optional artifact extraction callback.
            step_result: Step execution result to augment.
            step_context: Runtime context used for callback-based extraction.

        Returns:
            Step result with normalized artifact manifests attached.
        """
        artifacts: list[WorkflowArtifact] = list(step_result.artifacts)
        artifacts.extend(
            self._coerce_artifacts_from_output(
                step_id=step_result.step_id,
                step_output=step_result.output,
            )
        )

        artifacts_builder = getattr(step, "artifacts_builder", None)
        if callable(artifacts_builder):
            callback_context = dict(step_context)
            callback_context["step_output"] = dict(step_result.output)
            callback_context["step_id"] = step_result.step_id
            try:
                # Builder callbacks can derive richer user-facing artifact manifests from full context.
                built_artifacts = artifacts_builder(callback_context)
            except Exception as exc:
                return WorkflowStepResult(
                    step_id=step_result.step_id,
                    status="failed",
                    success=False,
                    output=dict(step_result.output),
                    error=f"Artifact builder failed: {exc}",
                    metadata={"stage": "artifact_builder"},
                    artifacts=tuple(artifacts),
                )
            artifacts.extend(
                self._normalize_artifact_entries(
                    step_id=step_result.step_id,
                    entries=built_artifacts,
                )
            )

        if not artifacts:
            return step_result

        unique_artifacts = dedupe_artifacts(artifacts)
        return WorkflowStepResult(
            step_id=step_result.step_id,
            status=step_result.status,
            success=step_result.success,
            output=dict(step_result.output),
            error=step_result.error,
            metadata=dict(step_result.metadata),
            artifacts=tuple(unique_artifacts),
        )

    def _coerce_artifacts_from_output(
        self,
        *,
        step_id: str,
        step_output: Mapping[str, object],
    ) -> list[WorkflowArtifact]:
        """Extract artifact manifests from normalized step output payload.

        Args:
            step_id: Owning step id.
            step_output: Step output mapping.

        Returns:
            Normalized artifact records discovered in payload.
        """
        candidates: list[object] = []
        tool_artifacts = step_output.get("artifacts")
        if isinstance(tool_artifacts, list):
            candidates.extend(tool_artifacts)

        result_payload = step_output.get("result")
        if isinstance(result_payload, Mapping):
            nested_artifacts = result_payload.get("artifacts")
            if isinstance(nested_artifacts, list):
                candidates.extend(nested_artifacts)

        output_artifacts = step_output.get("output")
        if isinstance(output_artifacts, Mapping):
            nested = output_artifacts.get("artifacts")
            if isinstance(nested, list):
                candidates.extend(nested)

        return self._normalize_artifact_entries(step_id=step_id, entries=candidates)

    def _normalize_artifact_entries(
        self,
        *,
        step_id: str,
        entries: Sequence[object],
    ) -> list[WorkflowArtifact]:
        """Normalize arbitrary artifact entries into typed records.

        Args:
            step_id: Owning step id.
            entries: Raw artifact entries from a step payload.

        Returns:
            Normalized artifact list.
        """
        normalized: list[WorkflowArtifact] = []
        for entry in entries:
            if isinstance(entry, WorkflowArtifact):
                if not entry.producer_step_id:
                    # Backfill producer metadata so artifact provenance is always traceable.
                    normalized.append(
                        WorkflowArtifact(
                            path=entry.path,
                            mime=entry.mime,
                            title=entry.title,
                            summary=entry.summary,
                            audience=entry.audience,
                            producer_step_id=step_id,
                            sources=entry.sources or (WorkflowArtifactSource(step_id=step_id, field="artifacts"),),
                            metadata=dict(entry.metadata),
                        )
                    )
                else:
                    normalized.append(entry)
                continue
            if not isinstance(entry, Mapping):
                continue
            path = str(entry.get("path", "")).strip()
            if not path:
                continue
            mime = str(entry.get("mime", "application/octet-stream"))
            metadata_raw = entry.get("metadata")
            metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
            source_step_raw = entry.get("producer_step_id")
            source_step = (
                str(source_step_raw).strip()
                if isinstance(source_step_raw, str) and source_step_raw.strip()
                else step_id
            )
            # Default producer step to current step unless explicitly overridden.
            source_field = entry.get("source_field")
            normalized.append(
                WorkflowArtifact(
                    path=path,
                    mime=mime,
                    title=str(entry["title"]) if "title" in entry else None,
                    summary=str(entry["summary"]) if "summary" in entry else None,
                    audience=str(entry["audience"]) if "audience" in entry else None,
                    producer_step_id=source_step,
                    sources=(
                        WorkflowArtifactSource(
                            step_id=source_step,
                            field=str(source_field) if isinstance(source_field, str) else None,
                        ),
                    ),
                    metadata=metadata,
                )
            )
        return normalized

    def _collect_artifacts(
        self,
        *,
        step_results: Mapping[str, WorkflowStepResult],
        execution_order: Sequence[str],
    ) -> list[WorkflowArtifact]:
        """Collect ordered unique artifacts across all step results.

        Args:
            step_results: Step results keyed by step id.
            execution_order: Step execution order.

        Returns:
            Ordered unique artifact list.
        """
        artifacts: list[WorkflowArtifact] = []
        for step_id in execution_order:
            step_result = step_results.get(step_id)
            if step_result is None:
                continue
            artifacts.extend(step_result.artifacts)
        return _dedupe_artifacts(artifacts)

    def _resolve_final_output(
        self,
        *,
        step_results: Mapping[str, WorkflowStepResult],
        execution_order: Sequence[str],
    ) -> object:
        """Select one canonical final output payload from step results.

        Args:
            step_results: Step results keyed by step id.
            execution_order: Step execution order.

        Returns:
            Canonical final output payload from the last successful completed step.
        """
        for step_id in reversed(execution_order):
            step_result = step_results.get(step_id)
            if step_result is None:
                continue
            if step_result.status == "completed" and step_result.success:
                if "final_output" in step_result.output:
                    return step_result.output["final_output"]
                return dict(step_result.output)
        return {}


def _dedupe_artifacts(artifacts: Sequence[WorkflowArtifact]) -> list[WorkflowArtifact]:
    """Deduplicate artifacts while preserving first-seen order.

    Args:
        artifacts: Raw artifact sequence.

    Returns:
        Deduplicated artifact list.
    """
    return dedupe_artifacts(artifacts)
