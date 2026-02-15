"""Runnable example showing one ``SingleStepDirectLLMAgent`` execution.

The script calls a local llama-cpp server by default, runs one prompt directly
through the model, and prints the resulting ``AgentResult`` payload.
"""

from design_research_agents import LlamaCppServerLLMClient
from design_research_agents.agent import SingleStepDirectLLMAgent


def main() -> None:
    """Execute one direct-LLM agent run and print structured output."""
    llm_client = LlamaCppServerLLMClient()
    agent = SingleStepDirectLLMAgent(llm_client=llm_client)

    result = agent.run(
        prompt="What is two plus two?",
        request_id="example-direct-llm-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
