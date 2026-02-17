"""Runnable example showing one ``SingleStepToolRouterAgent`` execution end-to-end.

The script calls a local llama-cpp server by default, builds runtime/tool dependencies, and
executes runtime-driven route selection with built-in default schemas.
"""

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepToolRouterAgent


def main() -> None:
    """Execute one tool-router-agent run and print structured ``AgentResult`` output.

    Demonstrates route selection plus downstream tool invocation in one call.
    """
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = SingleStepToolRouterAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    # SingleStepToolRouterAgent derives available routes from ToolRuntime.list_tools().
    result = agent.run(
        prompt="Select which tool to provide a short status summary for this repository.",
        request_id="example-router-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
