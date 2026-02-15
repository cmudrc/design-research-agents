"""Reusable ``propose_critic`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from design_research_agents.agent import AgentRuntime
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolRuntime


class ProposeAndCritiqueOrchestrator(Agent):
    """Configured orchestrator chunk for ``propose_critic`` mode."""

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
            mode="propose_critic",
            controls=controls,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one propose-and-critique orchestration run."""
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


def propose_and_critique(
    *,
    llm_client: LLMClient,
    tool_runtime: ToolRuntime,
    controls: RuntimeControls | None = None,
) -> ProposeAndCritiqueOrchestrator:
    """Return a configured ``propose_critic`` orchestrator chunk."""
    return ProposeAndCritiqueOrchestrator(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        controls=controls,
    )


__all__ = [
    "ProposeAndCritiqueOrchestrator",
    "propose_and_critique",
]
