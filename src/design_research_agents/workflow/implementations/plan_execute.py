"""Reusable ``plan_execute`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from uuid import uuid4

from design_research_agents.agent import AgentRuntime
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tracing import Tracer


class PlanExecuteWorkflow(Agent):
    """Configured workflow chunk for ``plan_execute`` mode."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        controls: RuntimeControls | None = None,
        plan_execute_planner_system_prompt: str | None = None,
        plan_execute_planner_user_prompt_template: str | None = None,
        plan_execute_executor_step_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize the underlying runtime."""
        self._default_request_id_prefix = _normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._runtime = AgentRuntime(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            mode="plan_execute",
            controls=controls,
            plan_execute_planner_system_prompt=plan_execute_planner_system_prompt,
            plan_execute_planner_user_prompt_template=plan_execute_planner_user_prompt_template,
            plan_execute_executor_step_prompt_template=plan_execute_executor_step_prompt_template,
            tracer=tracer,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one plan-execute orchestration run."""
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        return self._runtime.run(
            prompt,
            request_id=resolved_request_id,
            dependencies=_merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            ),
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute one run and emit streaming events."""
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        yield from self._runtime.run_stream(
            prompt,
            request_id=resolved_request_id,
            dependencies=_merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            ),
        )


def _merge_dependencies(
    *,
    default_dependencies: Mapping[str, object],
    run_dependencies: Mapping[str, object] | None,
) -> dict[str, object]:
    merged = dict(default_dependencies)
    if run_dependencies is not None:
        merged.update(run_dependencies)
    return merged


def _normalize_request_id_prefix(default_request_id_prefix: str | None) -> str | None:
    if default_request_id_prefix is None:
        return None
    normalized = default_request_id_prefix.strip()
    if not normalized:
        raise ValueError("default_request_id_prefix must be non-empty when provided.")
    return normalized


def _resolve_request_id(*, request_id: str | None, default_prefix: str | None) -> str | None:
    if request_id is not None and request_id.strip():
        return request_id
    if default_prefix is None:
        return request_id
    return f"{default_prefix}:{uuid4().hex}"


__all__ = [
    "PlanExecuteWorkflow",
]
