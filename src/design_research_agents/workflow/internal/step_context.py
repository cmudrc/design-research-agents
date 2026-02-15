"""Step context and routing helpers for workflow execution."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

from design_research_agents.contracts.workflow import (
    AgentStep,
    LogicStep,
    ToolStep,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowStepResult,
)


def build_step_context(
    *,
    base_context: Mapping[str, object],
    step_id: str,
    step_dependencies: Sequence[str],
    step_results: Mapping[str, WorkflowStepResult],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
) -> dict[str, object]:
    """Build per-step context including normalized dependency result payloads."""
    dependency_results: dict[str, dict[str, object]] = {}
    for dependency in step_dependencies:
        dependency_result = step_results.get(dependency)
        if dependency_result is None:
            continue
        dependency_results[dependency] = {
            "status": dependency_result.status,
            "success": dependency_result.success,
            "output": dict(dependency_result.output),
            "error": dependency_result.error,
            "metadata": dict(dependency_result.metadata),
        }

    context = dict(base_context)
    context["dependency_results"] = dependency_results
    context["_workflow"] = {
        "request_id": request_id,
        "step_id": step_id,
        "execution_mode": execution_mode,
        "failure_policy": failure_policy,
        "dependency_count": len(step_dependencies),
    }
    return context


def build_invocation_dependencies(
    *,
    base_dependencies: Mapping[str, object],
    step_id: str,
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    step_context: Mapping[str, object],
) -> dict[str, object]:
    """Build dependency payload passed to tool or agent invocation."""
    invocation_dependencies = dict(base_dependencies)
    raw_dependency_results = step_context.get("dependency_results")
    dependency_results = (
        dict(raw_dependency_results) if isinstance(raw_dependency_results, Mapping) else {}
    )
    invocation_dependencies["_workflow"] = {
        "request_id": request_id,
        "step_id": step_id,
        "execution_mode": execution_mode,
        "failure_policy": failure_policy,
        "dependency_results": dependency_results,
    }
    return invocation_dependencies


def resolve_tool_input(*, step: ToolStep, step_context: Mapping[str, object]) -> dict[str, object]:
    """Resolve one tool input payload from builder callback or static data."""
    if step.input_builder is not None:
        built_input = step.input_builder(step_context)
        if not isinstance(built_input, Mapping):
            raise TypeError("ToolStep input_builder must return a mapping.")
        return dict(built_input)
    if step.input_data is None:
        return {}
    return dict(step.input_data)


def resolve_agent_prompt(*, step: AgentStep, step_context: Mapping[str, object]) -> str:
    """Resolve one agent prompt from builder callback, static prompt, or context fallback."""
    if step.prompt_builder is not None:
        built_prompt = step.prompt_builder(step_context)
        if not isinstance(built_prompt, str):
            raise TypeError("AgentStep prompt_builder must return a string.")
        normalized_prompt = built_prompt.strip()
        if not normalized_prompt:
            raise ValueError("AgentStep prompt_builder returned an empty prompt.")
        return normalized_prompt

    if isinstance(step.prompt, str) and step.prompt.strip():
        return step.prompt.strip()

    fallback_prompt = step_context.get("prompt")
    if isinstance(fallback_prompt, str) and fallback_prompt.strip():
        return fallback_prompt.strip()

    raise ValueError(f"AgentStep '{step.step_id}' requires a non-empty prompt or prompt_builder.")


def has_upstream_failure(
    *,
    dependencies: Sequence[str],
    step_results: Mapping[str, WorkflowStepResult],
) -> bool:
    """Return true when any dependency has already failed."""
    for dependency in dependencies:
        dependency_result = step_results.get(dependency)
        if dependency_result is None:
            continue
        if not dependency_result.success:
            return True
    return False


def route_deactivations(
    *,
    step: LogicStep,
    step_output: Mapping[str, object],
    dependents: Mapping[str, Sequence[str]],
) -> tuple[set[str], str | None]:
    """Resolve branch deactivations for logic steps that emit route decisions."""
    if step.route_map is None:
        return set(), None

    route_map = normalize_route_map(step.route_map)
    if not route_map:
        return set(), (
            f"Step '{step.step_id}' declared route_map but no valid routes were configured."
        )

    route_value = step_output.get("route")
    if not isinstance(route_value, str) or not route_value.strip():
        return set(), (
            f"Step '{step.step_id}' declared route_map but output did not include a non-empty "
            "'route' string."
        )

    selected_route = route_value.strip()
    selected_targets = set(route_map.get(selected_route, ()))
    if not selected_targets:
        return set(), (
            f"Step '{step.step_id}' selected route '{selected_route}' but no targets were "
            "configured for that route."
        )

    all_targets = {target for targets in route_map.values() for target in targets}
    deactivated_steps: set[str] = set()
    for non_selected_target in all_targets.difference(selected_targets):
        for descendant in _collect_descendants(
            start_step=non_selected_target,
            dependents=dependents,
        ):
            if descendant not in selected_targets:
                deactivated_steps.add(descendant)

    return deactivated_steps, None


def normalize_route_map(
    raw_route_map: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Normalize and filter route-map keys and target ids."""
    normalized: dict[str, tuple[str, ...]] = {}
    for route_key, targets in raw_route_map.items():
        if not isinstance(route_key, str):
            continue
        normalized_route = route_key.strip()
        if not normalized_route:
            continue
        normalized_targets = tuple(
            target.strip() for target in targets if isinstance(target, str) and target.strip()
        )
        if normalized_targets:
            normalized[normalized_route] = normalized_targets
    return normalized


def _collect_descendants(
    *,
    start_step: str,
    dependents: Mapping[str, Sequence[str]],
) -> set[str]:
    descendants: set[str] = set()
    queue: deque[str] = deque([start_step])
    while queue:
        current_step = queue.popleft()
        if current_step in descendants:
            continue
        descendants.add(current_step)
        for child in dependents.get(current_step, ()):  # pragma: no branch - tiny helper
            queue.append(child)
    return descendants
