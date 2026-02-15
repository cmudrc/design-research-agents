"""Sequential workflow orchestrator implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.agent._run_options import normalize_dependencies, resolve_request_id
from design_research_agents.contracts.orchestrator import (
    Orchestrator,
    WorkflowFailurePolicy,
    WorkflowNode,
    WorkflowNodeResult,
    WorkflowResult,
)
from design_research_agents.orchestrator._shared import (
    build_node_context,
    prepare_workflow_graph,
    should_skip_for_upstream_failure,
    try_validate_output,
    validate_node_input,
)
from design_research_agents.schema import SchemaValidationError
from design_research_agents.tracing import finish_trace_run, start_trace_run
from design_research_agents.tracing.context import current_span_id, current_trace_session


class SequentialOrchestrator(Orchestrator):
    """Execute workflow nodes in declared order with dependency checks."""

    def run(
        self,
        nodes: Sequence[WorkflowNode],
        *,
        context: Mapping[str, object] | None = None,
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute nodes sequentially and return a workflow result."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        base_context = dict(context or {})
        trace_scope = start_trace_run(
            agent_name="SequentialOrchestrator",
            request_id=resolved_request_id,
            input_payload={
                "node_count": len(nodes),
                "failure_policy": failure_policy,
            },
            dependencies=resolved_dependencies,
        )

        try:
            prepared = prepare_workflow_graph(nodes)
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        node_results: dict[str, WorkflowNodeResult] = {}
        execution_order: list[str] = []

        for node in nodes:
            node_id = node.node_id.strip()
            dependencies_missing = [
                dependency
                for dependency in node.dependencies
                if isinstance(dependency, str) and dependency.strip() not in node_results
            ]
            if dependencies_missing:
                error = (
                    f"Node '{node_id}' cannot run before dependencies are resolved: "
                    f"{', '.join(sorted(dependencies_missing))}."
                )
                finish_trace_run(trace_scope, error=error)
                raise ValueError(error)

            if failure_policy == "skip_dependents" and should_skip_for_upstream_failure(
                node=node,
                node_results=node_results,
            ):
                node_results[node_id] = WorkflowNodeResult(
                    node_id=node_id,
                    status="skipped",
                    success=False,
                    output={},
                    error="skipped_upstream_failure",
                )
                execution_order.append(node_id)
                continue

            node_context = build_node_context(
                base_context=base_context,
                node=node,
                node_results=node_results,
            )

            try:
                validate_node_input(node=node, context=node_context)
            except SchemaValidationError as exc:
                node_results[node_id] = WorkflowNodeResult(
                    node_id=node_id,
                    status="failed",
                    success=False,
                    output={},
                    error=str(exc),
                    metadata={"stage": "input_validation"},
                )
                execution_order.append(node_id)
                continue

            node_span_id = _start_node_span(node=node)
            try:
                output = dict(node.run(node_context))
            except Exception as exc:
                _finish_node_span(
                    span_id=node_span_id,
                    node_id=node_id,
                    status="failed",
                    error=str(exc),
                )
                node_results[node_id] = WorkflowNodeResult(
                    node_id=node_id,
                    status="failed",
                    success=False,
                    output={},
                    error=str(exc),
                    metadata={"stage": "execution"},
                )
                execution_order.append(node_id)
                continue

            output_validation_error = try_validate_output(node=node, output=output)
            if output_validation_error is not None:
                _finish_node_span(
                    span_id=node_span_id,
                    node_id=node_id,
                    status="failed",
                    error=output_validation_error,
                )
                node_results[node_id] = WorkflowNodeResult(
                    node_id=node_id,
                    status="failed",
                    success=False,
                    output=output,
                    error=output_validation_error,
                    metadata={"stage": "output_validation"},
                )
                execution_order.append(node_id)
                continue

            _finish_node_span(
                span_id=node_span_id,
                node_id=node_id,
                status="completed",
                error=None,
            )
            node_results[node_id] = WorkflowNodeResult(
                node_id=node_id,
                status="completed",
                success=True,
                output=output,
            )
            execution_order.append(node_id)

        success = all(result.success for result in node_results.values())
        workflow_result = WorkflowResult(
            success=success,
            node_results=node_results,
            execution_order=execution_order,
            metadata={
                "orchestrator": "sequential",
                "failure_policy": failure_policy,
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "node_count": len(nodes),
                "validated_node_count": len(prepared.node_map),
            },
        )
        finish_trace_run(trace_scope, result=workflow_result)
        return workflow_result


def _start_node_span(*, node: WorkflowNode) -> str | None:
    session = current_trace_session()
    if session is None:
        return None
    parent_span = current_span_id() or session.root_span_id
    return session.start_span(
        "WorkflowNodeStarted",
        parent_span_id=parent_span,
        attributes={
            "node_id": node.node_id,
            "dependencies": list(node.dependencies),
        },
    )


def _finish_node_span(
    *,
    span_id: str | None,
    node_id: str,
    status: str,
    error: str | None,
) -> None:
    session = current_trace_session()
    if session is None or span_id is None:
        return
    session.finish_span(
        "WorkflowNodeFinished",
        span_id=span_id,
        attributes={
            "node_id": node_id,
            "status": status,
            "error": error,
        },
    )
