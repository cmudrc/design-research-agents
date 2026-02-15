"""Shared helpers for orchestrator implementations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.orchestrator import WorkflowNode, WorkflowNodeResult
from design_research_agents.schema import SchemaValidationError, validate_payload_against_schema


@dataclass(slots=True, frozen=True)
class PreparedWorkflow:
    """Normalized workflow graph artifacts."""

    node_map: dict[str, WorkflowNode]
    dependents: dict[str, list[str]]


def prepare_workflow_graph(nodes: Sequence[WorkflowNode]) -> PreparedWorkflow:
    """Build and validate static graph artifacts from workflow nodes."""
    node_map: dict[str, WorkflowNode] = {}
    for node in nodes:
        node_id = _normalize_node_id(getattr(node, "node_id", None))
        if node_id in node_map:
            raise ValueError(f"Duplicate workflow node id '{node_id}'.")
        node_map[node_id] = node

    dependents: dict[str, list[str]] = defaultdict(list)
    for node_id, node in node_map.items():
        for dependency in _normalize_dependencies(node):
            if dependency not in node_map:
                raise ValueError(f"Node '{node_id}' depends on unknown node '{dependency}'.")
            dependents[dependency].append(node_id)
    return PreparedWorkflow(node_map=node_map, dependents=dict(dependents))


def build_node_context(
    *,
    base_context: Mapping[str, object],
    node: WorkflowNode,
    node_results: Mapping[str, WorkflowNodeResult],
) -> dict[str, object]:
    """Build per-node execution context with dependency result injection."""
    context = dict(base_context)
    dependency_results: dict[str, dict[str, object]] = {}
    for dependency in _normalize_dependencies(node):
        dependency_result = node_results.get(dependency)
        if dependency_result is None:
            continue
        dependency_results[dependency] = {
            "status": dependency_result.status,
            "success": dependency_result.success,
            "output": dict(dependency_result.output),
            "error": dependency_result.error,
            "metadata": dict(dependency_result.metadata),
        }
    context["dependency_results"] = dependency_results
    return context


def validate_node_input(*, node: WorkflowNode, context: Mapping[str, object]) -> None:
    """Validate node input context against its input schema."""
    validate_payload_against_schema(
        payload=dict(context),
        schema=node.input_schema,
        location=f"node[{node.node_id}].input",
    )


def validate_node_output(*, node: WorkflowNode, output: Mapping[str, object]) -> None:
    """Validate node output against its output schema."""
    validate_payload_against_schema(
        payload=dict(output),
        schema=node.output_schema,
        location=f"node[{node.node_id}].output",
    )


def should_skip_for_upstream_failure(
    *,
    node: WorkflowNode,
    node_results: Mapping[str, WorkflowNodeResult],
) -> bool:
    """Return whether node should skip due to failed/skipped dependencies."""
    for dependency in _normalize_dependencies(node):
        dependency_result = node_results.get(dependency)
        if dependency_result is None:
            continue
        if not dependency_result.success:
            return True
    return False


def _normalize_node_id(raw_node_id: object) -> str:
    if not isinstance(raw_node_id, str):
        raise ValueError("Workflow node_id must be a non-empty string.")
    normalized = raw_node_id.strip()
    if not normalized:
        raise ValueError("Workflow node_id must be a non-empty string.")
    return normalized


def _normalize_dependencies(node: WorkflowNode) -> tuple[str, ...]:
    raw_dependencies = getattr(node, "dependencies", ())
    if not isinstance(raw_dependencies, Sequence) or isinstance(raw_dependencies, (str, bytes)):
        return ()
    normalized: list[str] = []
    for dependency in raw_dependencies:
        if not isinstance(dependency, str):
            continue
        stripped = dependency.strip()
        if stripped:
            normalized.append(stripped)
    return tuple(normalized)


def try_validate_output(
    *,
    node: WorkflowNode,
    output: Mapping[str, object],
) -> str | None:
    """Validate output and return string error instead of raising."""
    try:
        validate_node_output(node=node, output=output)
    except SchemaValidationError as exc:
        return str(exc)
    return None
