"""Runnable example showing one ``RouterAgent`` execution end-to-end.

The script configures a local backend, builds runtime/tool dependencies, and
executes runtime-driven route selection with built-in default schemas.
"""

import design_research_agents


def main() -> None:
    """Execute one router-agent run and print structured ``AgentResult`` output.

    Demonstrates route selection plus downstream tool invocation in one call.
    """
    llm_client = design_research_agents.create_default_llm_client()
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.RouterAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    # RouterAgent will derive available routes from ToolRuntime.list_tools().
    result = agent.run(
        input="Select which tool to provide a short status summary for this repository.",
        request_id="example-router-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
