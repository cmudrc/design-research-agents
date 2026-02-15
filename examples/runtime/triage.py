"""Runnable example for ``AgentRuntime`` in ``triage`` mode."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from _runtime_example_support import SequenceResponseLLMClient

import design_research_agents
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent


class _StaticDelegatedAgent(Agent):
    """Small deterministic delegated agent for triage demos."""

    def __init__(self, *, label: str) -> None:
        self._label = label

    def run(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del input, request_id, dependencies
        return AgentResult(
            output={"selected_delegate": self._label},
            success=True,
            tool_results=[],
            metadata={"delegate": self._label},
        )

    def run_stream(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        yield AgentStreamEvent(
            kind="completed",
            result=self.run(
                input,
                request_id=request_id,
                dependencies=dependencies,
            ),
        )


def main() -> None:
    llm_client = SequenceResponseLLMClient(
        response_texts=['{"selection": "stats_agent", "reason": "best fit"}']
    )

    agent = design_research_agents.AgentRuntime(
        llm_client=llm_client,
        tool_runtime=design_research_agents.BaseToolRuntime(),
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
