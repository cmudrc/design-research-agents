"""Reusable ``plan_execute`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from design_research_agents.agent import AgentRuntime
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolRuntime


class PlanExecuteOrchestrator(Agent):
    """Configured orchestrator chunk for ``plan_execute`` mode."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        controls: RuntimeControls | None = None,
    ) -> None:
        """Store dependencies and initialize the underlying runtime."""
        self._runtime = AgentRuntime(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            mode="plan_execute",
            controls=controls,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one plan-execute orchestration run."""
        return self._runtime.run(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute one run and emit streaming events."""
        yield from self._runtime.run_stream(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )


def plan_and_execute(
    *,
    llm_client: LLMClient,
    tool_runtime: ToolRuntime,
    controls: RuntimeControls | None = None,
) -> PlanExecuteOrchestrator:
    """Return a configured ``plan_execute`` orchestrator chunk."""
    return PlanExecuteOrchestrator(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        controls=controls,
    )


__all__ = [
    "PlanExecuteOrchestrator",
    "plan_and_execute",
]
