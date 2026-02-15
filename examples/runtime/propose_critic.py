"""Runnable example for ``AgentRuntime`` in ``propose_critic`` mode."""

from __future__ import annotations

import design_research_agents as dra


def main() -> None:
    """Run the ``propose_critic`` runtime example."""
    llm_client = dra.llm.create_default_llm_client()

    agent = dra.agents.AgentRuntime(
        llm_client=llm_client,
        tool_runtime=dra.tools.UnifiedToolRuntime(),
        mode="propose_critic",
    )
    result = agent.run("Write a short design summary for this repository.")
    print(result)


if __name__ == "__main__":
    main()
