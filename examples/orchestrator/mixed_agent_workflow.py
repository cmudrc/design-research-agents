"""Runnable example for the reusable mixed workflow orchestration chunk."""

import design_research_agents as dra
from design_research_agents.orchestrator.implementations.mixed_agent_workflow import (
    mixed_agent_workflow,
)


def main() -> None:
    """Run the configured mixed workflow and print the aggregated result."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    orchestrator = mixed_agent_workflow(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )
    result = orchestrator.run()
    print(result)


if __name__ == "__main__":
    main()
