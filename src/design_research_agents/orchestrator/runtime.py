"""Deterministic workflow runtime for tool, agent, and logic steps."""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import cast

from design_research_agents.agent._run_options import normalize_dependencies, resolve_request_id
from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.orchestrator import (
    AgentStep,
    LogicStep,
    ToolStep,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowResult,
    WorkflowRunner,
    WorkflowStep,
    WorkflowStepResult,
)
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tracing import finish_trace_run, start_trace_run
from design_research_agents.tracing.context import current_span_id, current_trace_session


@dataclass(slots=True, frozen=True)
class _PreparedWorkflow:
    step_map: dict[str, WorkflowStep]
    dependencies: dict[str, tuple[str, ...]]
    dependents: dict[str, list[str]]


class WorkflowRuntime(WorkflowRunner):
    """Execute typed workflows with deterministic sequential or DAG scheduling."""

    def __init__(
        self,
        *,
        tool_runtime: ToolRuntime | None = None,
        agents: Mapping[str, Agent] | None = None,
    ) -> None:
        self._tool_runtime = tool_runtime
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
        )

        try:
            prepared = _prepare_workflow_graph(steps)
            if execution_mode == "dag":
                _validate_no_cycles(prepared.step_map, prepared.dependencies)

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
        prepared: _PreparedWorkflow,
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
            step_id = _normalize_step_id(step.step_id)
            step_dependencies = prepared.dependencies[step_id]

            unresolved_dependencies = [
                dependency for dependency in step_dependencies if dependency not in step_results
            ]
            if unresolved_dependencies:
                raise ValueError(
                    f"Step '{step_id}' cannot run before dependencies are resolved: "
                    f"{', '.join(sorted(unresolved_dependencies))}."
                )

            if step_id in deactivated_steps:
                step_results[step_id] = WorkflowStepResult(
                    step_id=step_id,
                    status="skipped",
                    success=False,
                    output={},
                    error="skipped_branch_not_selected",
                )
                execution_order.append(step_id)
                continue

            if failure_policy == "skip_dependents" and _has_upstream_failure(
                dependencies=step_dependencies,
                step_results=step_results,
            ):
                step_results[step_id] = WorkflowStepResult(
                    step_id=step_id,
                    status="skipped",
                    success=False,
                    output={},
                    error="skipped_upstream_failure",
                )
                execution_order.append(step_id)
                continue

            step_context = _build_step_context(
                base_context=base_context,
                step_id=step_id,
                step_dependencies=step_dependencies,
                step_results=step_results,
                request_id=resolved_request_id,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
            )
            step_result = self._execute_step(
                step=step,
                step_id=step_id,
                step_context=step_context,
                request_id=resolved_request_id,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
                dependencies=resolved_dependencies,
            )

            if step_result.success and isinstance(step, LogicStep):
                deactivation_update, route_error = _route_deactivations(
                    step=step,
                    step_output=step_result.output,
                    dependents=prepared.dependents,
                )
                if route_error is not None:
                    step_result = WorkflowStepResult(
                        step_id=step_id,
                        status="failed",
                        success=False,
                        output=dict(step_result.output),
                        error=route_error,
                        metadata={"stage": "routing"},
                    )
                else:
                    deactivated_steps.update(deactivation_update)

            step_results[step_id] = step_result
            execution_order.append(step_id)

        return step_results, execution_order

    def _run_dag(
        self,
        *,
        prepared: _PreparedWorkflow,
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

            if step_id in deactivated_steps:
                step_results[step_id] = WorkflowStepResult(
                    step_id=step_id,
                    status="skipped",
                    success=False,
                    output={},
                    error="skipped_branch_not_selected",
                )
                execution_order.append(step_id)
                _release_dependents(
                    step_id=step_id,
                    dependents=prepared.dependents,
                    in_degree=in_degree,
                    ready_steps=ready_steps,
                )
                continue

            if failure_policy == "skip_dependents" and _has_upstream_failure(
                dependencies=step_dependencies,
                step_results=step_results,
            ):
                step_results[step_id] = WorkflowStepResult(
                    step_id=step_id,
                    status="skipped",
                    success=False,
                    output={},
                    error="skipped_upstream_failure",
                )
                execution_order.append(step_id)
                _release_dependents(
                    step_id=step_id,
                    dependents=prepared.dependents,
                    in_degree=in_degree,
                    ready_steps=ready_steps,
                )
                continue

            step_context = _build_step_context(
                base_context=base_context,
                step_id=step_id,
                step_dependencies=step_dependencies,
                step_results=step_results,
                request_id=resolved_request_id,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
            )
            step_result = self._execute_step(
                step=step,
                step_id=step_id,
                step_context=step_context,
                request_id=resolved_request_id,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
                dependencies=resolved_dependencies,
            )

            if step_result.success and isinstance(step, LogicStep):
                deactivation_update, route_error = _route_deactivations(
                    step=step,
                    step_output=step_result.output,
                    dependents=prepared.dependents,
                )
                if route_error is not None:
                    step_result = WorkflowStepResult(
                        step_id=step_id,
                        status="failed",
                        success=False,
                        output=dict(step_result.output),
                        error=route_error,
                        metadata={"stage": "routing"},
                    )
                else:
                    deactivated_steps.update(deactivation_update)

            step_results[step_id] = step_result
            execution_order.append(step_id)
            _release_dependents(
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
        step_span_id = _start_step_span(step=step, step_id=step_id)
        with _activate_step_span(step_span_id):
            if isinstance(step, ToolStep):
                result = self._run_tool_step(
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                    request_id=request_id,
                    execution_mode=execution_mode,
                    failure_policy=failure_policy,
                    dependencies=dependencies,
                )
            elif isinstance(step, AgentStep):
                result = self._run_agent_step(
                    step=step,
                    step_id=step_id,
                    step_context=step_context,
                    request_id=request_id,
                    execution_mode=execution_mode,
                    failure_policy=failure_policy,
                    dependencies=dependencies,
                )
            else:
                result = self._run_logic_step(step=step, step_id=step_id, step_context=step_context)

        _finish_step_span(
            span_id=step_span_id,
            step_id=step_id,
            status=result.status,
            error=result.error,
        )
        return result

    def _run_tool_step(
        self,
        *,
        step: ToolStep,
        step_id: str,
        step_context: Mapping[str, object],
        request_id: str,
        execution_mode: WorkflowExecutionMode,
        failure_policy: WorkflowFailurePolicy,
        dependencies: Mapping[str, object],
    ) -> WorkflowStepResult:
        if self._tool_runtime is None:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error="Tool step requires a configured tool_runtime.",
                metadata={"stage": "tool_binding", "tool_name": step.tool_name},
            )

        available_tools = {tool_spec.name for tool_spec in self._tool_runtime.list_tools()}
        if step.tool_name not in available_tools:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error=f"Unknown tool '{step.tool_name}'.",
                metadata={"stage": "tool_binding", "tool_name": step.tool_name},
            )

        try:
            tool_input = _resolve_tool_input(step=step, step_context=step_context)
        except Exception as exc:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error=str(exc),
                metadata={"stage": "input_build", "tool_name": step.tool_name},
            )

        invocation_dependencies = _build_invocation_dependencies(
            base_dependencies=dependencies,
            step_id=step_id,
            request_id=request_id,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            step_context=step_context,
        )

        try:
            tool_result = self._tool_runtime.invoke(
                step.tool_name,
                tool_input,
                request_id=f"{request_id}:workflow:{step_id}",
                dependencies=invocation_dependencies,
            )
        except Exception as exc:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error=str(exc),
                metadata={"stage": "execution", "tool_name": step.tool_name},
            )

        serialized_output = asdict(tool_result)
        if not tool_result.success:
            tool_error_message = (
                tool_result.error.message
                if tool_result.error is not None
                else "Tool invocation failed."
            )
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output=serialized_output,
                error=tool_error_message,
                metadata={"stage": "execution", "tool_name": step.tool_name},
            )

        return WorkflowStepResult(
            step_id=step_id,
            status="completed",
            success=True,
            output=serialized_output,
            metadata={"stage": "execution", "tool_name": step.tool_name},
        )

    def _run_agent_step(
        self,
        *,
        step: AgentStep,
        step_id: str,
        step_context: Mapping[str, object],
        request_id: str,
        execution_mode: WorkflowExecutionMode,
        failure_policy: WorkflowFailurePolicy,
        dependencies: Mapping[str, object],
    ) -> WorkflowStepResult:
        selected_agent = self._agents.get(step.agent_name)
        if selected_agent is None:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error=f"Unknown agent '{step.agent_name}'.",
                metadata={"stage": "agent_binding", "agent_name": step.agent_name},
            )

        try:
            prompt = _resolve_agent_prompt(step=step, step_context=step_context)
        except Exception as exc:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error=str(exc),
                metadata={"stage": "input_build", "agent_name": step.agent_name},
            )

        invocation_dependencies = _build_invocation_dependencies(
            base_dependencies=dependencies,
            step_id=step_id,
            request_id=request_id,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            step_context=step_context,
        )

        try:
            agent_result = selected_agent.run(
                prompt,
                request_id=f"{request_id}:workflow:{step_id}",
                dependencies=invocation_dependencies,
            )
        except Exception as exc:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error=str(exc),
                metadata={"stage": "execution", "agent_name": step.agent_name},
            )

        serialized_output = agent_result.asdict()
        if not agent_result.success:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output=serialized_output,
                error=str(agent_result.output.get("error", "Agent execution failed.")),
                metadata={"stage": "execution", "agent_name": step.agent_name},
            )

        return WorkflowStepResult(
            step_id=step_id,
            status="completed",
            success=True,
            output=serialized_output,
            metadata={"stage": "execution", "agent_name": step.agent_name},
        )

    def _run_logic_step(
        self,
        *,
        step: LogicStep,
        step_id: str,
        step_context: Mapping[str, object],
    ) -> WorkflowStepResult:
        try:
            step_output = dict(step.handler(step_context))
        except Exception as exc:
            return WorkflowStepResult(
                step_id=step_id,
                status="failed",
                success=False,
                output={},
                error=str(exc),
                metadata={"stage": "execution"},
            )

        return WorkflowStepResult(
            step_id=step_id,
            status="completed",
            success=True,
            output=step_output,
            metadata={"stage": "execution"},
        )


def _prepare_workflow_graph(steps: Sequence[WorkflowStep]) -> _PreparedWorkflow:
    step_map: dict[str, WorkflowStep] = {}
    dependencies: dict[str, tuple[str, ...]] = {}

    for step in steps:
        step_id = _normalize_step_id(step.step_id)
        if step_id in step_map:
            raise ValueError(f"Duplicate workflow step id '{step_id}'.")
        step_map[step_id] = step
        dependencies[step_id] = _normalize_dependencies(step.dependencies)

    dependents: dict[str, list[str]] = defaultdict(list)
    for step_id, step_dependencies in dependencies.items():
        for dependency in step_dependencies:
            if dependency not in step_map:
                raise ValueError(f"Step '{step_id}' depends on unknown step '{dependency}'.")
            dependents[dependency].append(step_id)

    return _PreparedWorkflow(
        step_map=step_map,
        dependencies=dependencies,
        dependents=dict(dependents),
    )


def _normalize_step_id(raw_step_id: object) -> str:
    if not isinstance(raw_step_id, str):
        raise ValueError("Workflow step_id must be a non-empty string.")
    normalized = raw_step_id.strip()
    if not normalized:
        raise ValueError("Workflow step_id must be a non-empty string.")
    return normalized


def _normalize_dependencies(raw_dependencies: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for dependency in raw_dependencies:
        if not isinstance(dependency, str):
            continue
        dependency_id = dependency.strip()
        if dependency_id:
            normalized.append(dependency_id)
    return tuple(normalized)


def _build_step_context(
    *,
    base_context: Mapping[str, object],
    step_id: str,
    step_dependencies: Sequence[str],
    step_results: Mapping[str, WorkflowStepResult],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
) -> dict[str, object]:
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


def _build_invocation_dependencies(
    *,
    base_dependencies: Mapping[str, object],
    step_id: str,
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    step_context: Mapping[str, object],
) -> dict[str, object]:
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


def _resolve_tool_input(*, step: ToolStep, step_context: Mapping[str, object]) -> dict[str, object]:
    if step.input_builder is not None:
        built_input = step.input_builder(step_context)
        if not isinstance(built_input, Mapping):
            raise TypeError("ToolStep input_builder must return a mapping.")
        return dict(built_input)
    if step.input_data is None:
        return {}
    return dict(step.input_data)


def _resolve_agent_prompt(*, step: AgentStep, step_context: Mapping[str, object]) -> str:
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


def _has_upstream_failure(
    *,
    dependencies: Sequence[str],
    step_results: Mapping[str, WorkflowStepResult],
) -> bool:
    for dependency in dependencies:
        dependency_result = step_results.get(dependency)
        if dependency_result is None:
            continue
        if not dependency_result.success:
            return True
    return False


def _route_deactivations(
    *,
    step: LogicStep,
    step_output: Mapping[str, object],
    dependents: Mapping[str, Sequence[str]],
) -> tuple[set[str], str | None]:
    if step.route_map is None:
        return set(), None

    route_map = _normalize_route_map(step.route_map)
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


def _normalize_route_map(
    raw_route_map: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
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


def _release_dependents(
    *,
    step_id: str,
    dependents: Mapping[str, Sequence[str]],
    in_degree: dict[str, int],
    ready_steps: list[str],
) -> None:
    for dependent in dependents.get(step_id, ()):  # pragma: no branch - tiny helper
        in_degree[dependent] = max(0, in_degree[dependent] - 1)
        if in_degree[dependent] == 0:
            heapq.heappush(ready_steps, dependent)


def _validate_no_cycles(
    step_map: Mapping[str, WorkflowStep],
    dependencies: Mapping[str, Sequence[str]],
) -> None:
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


def _step_kind(step: WorkflowStep) -> str:
    if isinstance(step, ToolStep):
        return "tool"
    if isinstance(step, AgentStep):
        return "agent"
    return "logic"


def _start_step_span(*, step: WorkflowStep, step_id: str) -> str | None:
    session = current_trace_session()
    if session is None:
        return None
    parent_span_id = current_span_id() or session.root_span_id
    return cast(
        str,
        session.start_span(
            "WorkflowStepStarted",
            parent_span_id=parent_span_id,
            attributes={
                "step_id": step_id,
                "step_type": _step_kind(step),
                "dependencies": list(step.dependencies),
            },
        ),
    )


def _finish_step_span(*, span_id: str | None, step_id: str, status: str, error: str | None) -> None:
    session = current_trace_session()
    if session is None or span_id is None:
        return
    session.finish_span(
        "WorkflowStepFinished",
        span_id=span_id,
        attributes={
            "step_id": step_id,
            "status": status,
            "error": error,
        },
    )


@contextmanager
def _activate_step_span(span_id: str | None) -> Iterator[None]:
    if span_id is None:
        yield
        return

    from design_research_agents.tracing import context as tracing_context

    token = tracing_context._CURRENT_SPAN.set(span_id)
    try:
        yield
    finally:
        tracing_context._CURRENT_SPAN.reset(token)
