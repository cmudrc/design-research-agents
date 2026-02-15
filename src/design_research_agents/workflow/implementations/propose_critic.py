"""Reusable ``propose_critic`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from design_research_agents.agent import AgentRuntime
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tracing import Tracer


class ProposeAndCritiqueWorkflow(Agent):
    """Configured workflow chunk for ``propose_critic`` mode."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        controls: RuntimeControls | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize the underlying runtime."""
        self._runtime = AgentRuntime(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            mode="propose_critic",
            controls=controls,
            tracer=tracer,
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


def propose_and_critique_workflow(
    *,
    llm_client: LLMClient,
    tool_runtime: ToolRuntime,
    controls: RuntimeControls | None = None,
    tracer: Tracer | None = None,
) -> ProposeAndCritiqueWorkflow:
    """Return a configured ``propose_critic`` workflow chunk."""
    return ProposeAndCritiqueWorkflow(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        controls=controls,
        tracer=tracer,
    )


__all__ = [
    "ProposeAndCritiqueWorkflow",
    "propose_and_critique_workflow",
]
