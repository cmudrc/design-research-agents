"""Runnable example for ``AgentRuntime`` in ``triage`` mode."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import design_research_agents as dra

Agent = dra.contracts.agent.Agent
AgentResult = dra.contracts.agent.AgentResult
AgentStreamEvent = dra.contracts.agent.AgentStreamEvent


class _StaticDelegatedAgent(Agent):
    """Small deterministic delegated agent for triage demos."""

    def __init__(self, *, label: str) -> None:
        self._label = label

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id, dependencies
        return AgentResult(
            output={"selected_delegate": self._label},
            success=True,
            tool_results=[],
            metadata={"delegate": self._label},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        yield AgentStreamEvent(
            kind="completed",
            result=self.run(
                prompt,
                request_id=request_id,
                dependencies=dependencies,
            ),
        )


def main() -> None:
    """Run the ``triage`` runtime example and print delegated output."""
    llm_client = dra.llm.create_default_llm_client()

    agent = dra.agents.AgentRuntime(
        llm_client=llm_client,
        tool_runtime=dra.tools.UnifiedToolRuntime(),
        mode="triage",
        triage_alternatives={
            "math_agent": _StaticDelegatedAgent(label="math_agent"),
            "stats_agent": _StaticDelegatedAgent(label="stats_agent"),
        },
    )
    result = agent.run("Count the words in this sentence.")
    print(result)


if __name__ == "__main__":
    main()
