"""Runnable example showing one ``MultiStepDirectLLMAgent`` execution lifecycle."""

from design_research_agents import LlamaCppServerLLMClient
from design_research_agents.agent import MultiStepDirectLLMAgent


def main() -> None:
    """Execute one multi-step direct-LLM run and print the resulting result."""
    llm_client = LlamaCppServerLLMClient()
    agent = MultiStepDirectLLMAgent(
        llm_client=llm_client,
        max_steps=3,
    )

    result = agent.run(
        prompt="Draft then finalize a concise answer to: what is 6 * 7?",
        request_id="example-multi-step-direct-llm-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
