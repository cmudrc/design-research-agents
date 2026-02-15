"""Runnable example for ``AgentRuntime`` in ``propose_critic`` mode."""

from __future__ import annotations

import json

from _runtime_example_support import SequenceResponseLLMClient

import design_research_agents


def main() -> None:
    """Run the ``propose_critic`` runtime example with fixed model outputs."""
    llm_client = SequenceResponseLLMClient(
        response_texts=[
            "Draft v1: simple proposal.",
            json.dumps(
                {
                    "approved": False,
                    "feedback": "Add more detail.",
                    "revision_goals": ["expand rationale"],
                }
            ),
            "Draft v2: proposal with more detail.",
            json.dumps(
                {
                    "approved": True,
                    "feedback": "Looks good.",
                    "revision_goals": [],
                }
            ),
        ]
    )

    agent = design_research_agents.AgentRuntime(
        llm_client=llm_client,
        tool_runtime=design_research_agents.BaseToolRuntime(),
        mode="propose_critic",
    )
    result = agent.run("Write a short design summary for this repository.")
    print(result)


if __name__ == "__main__":
    main()
