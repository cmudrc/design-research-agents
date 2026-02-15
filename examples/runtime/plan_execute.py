"""Runnable example for ``AgentRuntime`` in ``plan_execute`` mode."""

from __future__ import annotations

import design_research_agents as dra


def main() -> None:
    """Run the ``plan_execute`` runtime example."""
    llm_client = dra.llm.create_default_llm_client()

    agent = dra.agents.AgentRuntime(
        llm_client=llm_client,
        tool_runtime=dra.tools.UnifiedToolRuntime(),
        mode="plan_execute",
    )
    result = agent.run("Create and analyze a tiny runtime tools inventory.")
    print(result)


if __name__ == "__main__":
    main()
