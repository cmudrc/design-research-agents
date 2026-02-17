"""Runnable example showing one ``MultiStepToolRouterAgent`` execution lifecycle."""

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import MultiStepToolRouterAgent


def main() -> None:
    """Execute one multi-step tool-router run and print the resulting result."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = MultiStepToolRouterAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    result = agent.run(
        prompt="Compute 12 * (4 + 1), then stop with a final structured output.",
        request_id="example-multi-step-tool-router-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
