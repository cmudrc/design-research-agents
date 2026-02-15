"""Runnable example for intent routing across concrete DRA agent delegates.

This flow uses ``agent_routing`` runtime mode under the hood, but presents a clearer
intent/agent-routing entrypoint and terminology.
"""

import design_research_agents as dra
from design_research_agents.orchestrator.implementations.agent_routing import (
    agent_routing_and_delegate,
)


def main() -> None:
    """Route one prompt to the best delegate agent and print the final result."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()

    direct_llm_agent = dra.agents.SingleStepDirectLLMAgent(llm_client=llm_client)
    json_tool_agent = dra.agents.SingleStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    orchestrator = agent_routing_and_delegate(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        alternatives={
            "direct_llm_agent": direct_llm_agent,
            "json_tool_agent": json_tool_agent,
        },
        alternative_descriptions={
            "direct_llm_agent": "Use for direct text-only responses with no runtime tools.",
            "json_tool_agent": "Use for requests that should invoke runtime tools.",
        },
    )

    # Internal routing is performed by SingleStepRouterAgent (tool-routing).
    result = orchestrator.run(
        prompt="Calculate this expression and return the numeric result: 12 * (4 + 1)",
        request_id="example-agent-routing-orchestrator-001",
    )
    print(result)


if __name__ == "__main__":
    main()
