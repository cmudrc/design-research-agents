"""Deterministic single-thread DAG workflow orchestrator."""

from __future__ import annotations

import heapq
from collections import deque
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


class DagOrchestrator(Orchestrator):
    """Execute workflow nodes with deterministic topological scheduling."""

    def run(
        self,
        nodes: Sequence[WorkflowNode],
        *,
        context: Mapping[str, object] | None = None,
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute a DAG workflow and return per-node execution results."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        base_context = dict(context or {})
        trace_scope = start_trace_run(
            agent_name="DagOrchestrator",
            request_id=resolved_request_id,
            input_payload={
                "node_count": len(nodes),
                "failure_policy": failure_policy,
            },
            dependencies=resolved_dependencies,
        )

        try:
            prepared = prepare_workflow_graph(nodes)
            _validate_no_cycles(prepared.node_map)
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        in_degree: dict[str, int] = {
            node_id: len([dep for dep in node.dependencies if isinstance(dep, str) and dep.strip()])
            for node_id, node in prepared.node_map.items()
        }
        ready: list[str] = [node_id for node_id, degree in in_degree.items() if degree == 0]
        heapq.heapify(ready)

        node_results: dict[str, WorkflowNodeResult] = {}
        execution_order: list[str] = []
        deactivated_nodes: set[str] = set()

        while ready:
            node_id = heapq.heappop(ready)
            if node_id in node_results:
                continue

            node = prepared.node_map[node_id]
            if node_id in deactivated_nodes:
                node_results[node_id] = WorkflowNodeResult(
                    node_id=node_id,
                    status="skipped",
                    success=False,
                    output={},
                    error="skipped_branch_not_selected",
                )
                execution_order.append(node_id)
                _release_dependents(
                    node_id=node_id,
                    in_degree=in_degree,
                    dependents=prepared.dependents,
                    ready=ready,
                )
                continue

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
                _release_dependents(
                    node_id=node_id,
                    in_degree=in_degree,
                    dependents=prepared.dependents,
                    ready=ready,
                )
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
                _release_dependents(
                    node_id=node_id,
                    in_degree=in_degree,
                    dependents=prepared.dependents,
                    ready=ready,
                )
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
                _release_dependents(
                    node_id=node_id,
                    in_degree=in_degree,
                    dependents=prepared.dependents,
                    ready=ready,
                )
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
                _release_dependents(
                    node_id=node_id,
                    in_degree=in_degree,
                    dependents=prepared.dependents,
                    ready=ready,
                )
                continue

            route_map = _extract_route_map(node)
            if route_map is not None:
                route_value = output.get("route")
                if not isinstance(route_value, str) or not route_value.strip():
                    route_error = (
                        f"Node '{node_id}' declared route_map but output did not include "
                        "a non-empty 'route' string."
                    )
                    _finish_node_span(
                        span_id=node_span_id,
                        node_id=node_id,
                        status="failed",
                        error=route_error,
                    )
                    node_results[node_id] = WorkflowNodeResult(
                        node_id=node_id,
                        status="failed",
                        success=False,
                        output=output,
                        error=route_error,
                        metadata={"stage": "routing"},
                    )
                    execution_order.append(node_id)
                    _release_dependents(
                        node_id=node_id,
                        in_degree=in_degree,
                        dependents=prepared.dependents,
                        ready=ready,
                    )
                    continue
                route_key = route_value.strip()
                selected_targets = set(route_map.get(route_key, ()))
                if not selected_targets:
                    route_error = (
                        f"Node '{node_id}' selected route '{route_key}' but no targets were "
                        "configured for that route."
                    )
                    _finish_node_span(
                        span_id=node_span_id,
                        node_id=node_id,
                        status="failed",
                        error=route_error,
                    )
                    node_results[node_id] = WorkflowNodeResult(
                        node_id=node_id,
                        status="failed",
                        success=False,
                        output=output,
                        error=route_error,
                        metadata={"stage": "routing"},
                    )
                    execution_order.append(node_id)
                    _release_dependents(
                        node_id=node_id,
                        in_degree=in_degree,
                        dependents=prepared.dependents,
                        ready=ready,
                    )
                    continue

                all_branch_targets = {
                    target for targets in route_map.values() for target in targets
                }
                for non_selected_target in all_branch_targets.difference(selected_targets):
                    for descendant in _collect_descendants(
                        start_node=non_selected_target,
                        dependents=prepared.dependents,
                    ):
                        if descendant not in selected_targets:
                            deactivated_nodes.add(descendant)

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
            _release_dependents(
                node_id=node_id,
                in_degree=in_degree,
                dependents=prepared.dependents,
                ready=ready,
            )

        if len(node_results) != len(prepared.node_map):
            unresolved = sorted(set(prepared.node_map).difference(node_results.keys()))
            error = (
                "DAG execution ended with unresolved nodes; check dependency graph or branch "
                f"configuration. Unresolved nodes: {', '.join(unresolved)}"
            )
            finish_trace_run(trace_scope, error=error)
            raise RuntimeError(error)

        success = all(
            result.success or result.error == "skipped_branch_not_selected"
            for result in node_results.values()
        )
        workflow_result = WorkflowResult(
            success=success,
            node_results=node_results,
            execution_order=execution_order,
            metadata={
                "orchestrator": "dag",
                "failure_policy": failure_policy,
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "node_count": len(nodes),
            },
        )
        finish_trace_run(trace_scope, result=workflow_result)
        return workflow_result


def _release_dependents(
    *,
    node_id: str,
    in_degree: dict[str, int],
    dependents: Mapping[str, Sequence[str]],
    ready: list[str],
) -> None:
    for dependent in dependents.get(node_id, ()):  # pragma: no branch - tiny helper
        in_degree[dependent] = max(0, in_degree[dependent] - 1)
        if in_degree[dependent] == 0:
            heapq.heappush(ready, dependent)


def _extract_route_map(node: WorkflowNode) -> dict[str, tuple[str, ...]] | None:
    raw_route_map = getattr(node, "route_map", None)
    if not isinstance(raw_route_map, Mapping):
        return None

    normalized: dict[str, tuple[str, ...]] = {}
    for route_key, targets in raw_route_map.items():
        if not isinstance(route_key, str):
            continue
        normalized_key = route_key.strip()
        if not normalized_key:
            continue
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            continue
        normalized_targets = tuple(
            target.strip() for target in targets if isinstance(target, str) and target.strip()
        )
        if normalized_targets:
            normalized[normalized_key] = normalized_targets
    return normalized or None


def _collect_descendants(*, start_node: str, dependents: Mapping[str, Sequence[str]]) -> set[str]:
    descendants: set[str] = set()
    queue: deque[str] = deque([start_node])
    while queue:
        current = queue.popleft()
        if current in descendants:
            continue
        descendants.add(current)
        for child in dependents.get(current, ()):  # pragma: no branch - tiny helper
            queue.append(child)
    return descendants


def _validate_no_cycles(node_map: Mapping[str, WorkflowNode]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(node_id: str, path: list[str]) -> None:
        if node_id in visiting:
            cycle_start = path.index(node_id)
            cycle_path = [*path[cycle_start:], node_id]
            raise ValueError("Cycle detected in DAG workflow: " + " -> ".join(cycle_path))
        if node_id in visited:
            return

        visiting.add(node_id)
        path.append(node_id)
        node = node_map[node_id]
        for dependency in node.dependencies:
            if not isinstance(dependency, str):
                continue
            normalized = dependency.strip()
            if not normalized:
                continue
            _dfs(normalized, path)
        path.pop()
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in sorted(node_map.keys()):
        _dfs(node_id, [])


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
