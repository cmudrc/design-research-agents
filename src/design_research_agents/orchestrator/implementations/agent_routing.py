"""Reusable intent/agent-routing orchestration chunk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from design_research_agents.agent import AgentRuntime
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolRuntime


class AgentRoutingOrchestrator(Agent):
    """Configured orchestrator chunk for runtime ``agent_routing`` mode."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        alternatives: Mapping[str, Agent],
        alternative_descriptions: Mapping[str, str] | None = None,
        controls: RuntimeControls | None = None,
    ) -> None:
        """Store dependencies and initialize the underlying runtime."""
        self._runtime = AgentRuntime(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            mode="agent_routing",
            controls=controls,
            agent_routing_alternatives=alternatives,
            agent_routing_descriptions=alternative_descriptions,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one intent-routing orchestration run."""
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


def agent_routing_and_delegate(
    *,
    llm_client: LLMClient,
    tool_runtime: ToolRuntime,
    alternatives: Mapping[str, Agent],
    alternative_descriptions: Mapping[str, str] | None = None,
    controls: RuntimeControls | None = None,
) -> AgentRoutingOrchestrator:
    """Return a configured agent-routing orchestrator chunk."""
    return AgentRoutingOrchestrator(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        alternatives=alternatives,
        alternative_descriptions=alternative_descriptions,
        controls=controls,
    )


__all__ = [
    "AgentRoutingOrchestrator",
    "agent_routing_and_delegate",
]
