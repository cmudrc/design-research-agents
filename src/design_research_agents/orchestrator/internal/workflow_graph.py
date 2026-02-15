"""Workflow graph normalization and dependency scheduling helpers."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.orchestrator import WorkflowStep


@dataclass(slots=True, frozen=True)
class PreparedWorkflow:
    """Normalized workflow graph and dependency maps."""

    step_map: dict[str, WorkflowStep]
    dependencies: dict[str, tuple[str, ...]]
    dependents: dict[str, list[str]]


def prepare_workflow_graph(steps: Sequence[WorkflowStep]) -> PreparedWorkflow:
    """Normalize workflow steps into dependency and dependent lookup maps."""
    step_map: dict[str, WorkflowStep] = {}
    dependencies: dict[str, tuple[str, ...]] = {}

    for step in steps:
        step_id = normalize_step_id(step.step_id)
        if step_id in step_map:
            raise ValueError(f"Duplicate workflow step id '{step_id}'.")
        step_map[step_id] = step
        dependencies[step_id] = normalize_dependencies(step.dependencies)

    dependents: dict[str, list[str]] = defaultdict(list)
    for step_id, step_dependencies in dependencies.items():
        for dependency in step_dependencies:
            if dependency not in step_map:
                raise ValueError(f"Step '{step_id}' depends on unknown step '{dependency}'.")
            dependents[dependency].append(step_id)

    return PreparedWorkflow(
        step_map=step_map,
        dependencies=dependencies,
        dependents=dict(dependents),
    )


def normalize_step_id(raw_step_id: object) -> str:
    """Validate and normalize one workflow step id."""
    if not isinstance(raw_step_id, str):
        raise ValueError("Workflow step_id must be a non-empty string.")
    normalized = raw_step_id.strip()
    if not normalized:
        raise ValueError("Workflow step_id must be a non-empty string.")
    return normalized


def normalize_dependencies(raw_dependencies: Sequence[str]) -> tuple[str, ...]:
    """Normalize a step dependency list into a tuple of non-empty ids."""
    normalized: list[str] = []
    for dependency in raw_dependencies:
        if not isinstance(dependency, str):
            continue
        dependency_id = dependency.strip()
        if dependency_id:
            normalized.append(dependency_id)
    return tuple(normalized)


def release_dependents(
    *,
    step_id: str,
    dependents: Mapping[str, Sequence[str]],
    in_degree: dict[str, int],
    ready_steps: list[str],
) -> None:
    """Release DAG dependents once one step has completed or skipped."""
    for dependent in dependents.get(step_id, ()):  # pragma: no branch - tiny helper
        in_degree[dependent] = max(0, in_degree[dependent] - 1)
        if in_degree[dependent] == 0:
            heapq.heappush(ready_steps, dependent)


def validate_no_cycles(
    step_map: Mapping[str, WorkflowStep],
    dependencies: Mapping[str, Sequence[str]],
) -> None:
    """Validate that workflow dependencies form an acyclic graph."""
    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(step_id: str, path: list[str]) -> None:
        if step_id in visiting:
            cycle_start = path.index(step_id)
            cycle_path = [*path[cycle_start:], step_id]
            raise ValueError("Cycle detected in DAG workflow: " + " -> ".join(cycle_path))
        if step_id in visited:
            return

        visiting.add(step_id)
        path.append(step_id)
        for dependency in dependencies.get(step_id, ()):  # pragma: no branch - tiny helper
            _dfs(dependency, path)
        path.pop()
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in sorted(step_map):
        _dfs(step_id, [])
