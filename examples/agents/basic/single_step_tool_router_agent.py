"""Runnable example showing one ``SingleStepToolRouterAgent`` execution end-to-end."""

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepToolRouterAgent


def main() -> None:
    """Execute one tool-router run and print structured ``ExecutionResult`` output."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = SingleStepToolRouterAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    result = agent.run(
        prompt="Select which tool can summarize this repository and run it.",
        request_id="example-tool-router-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
