"""Runnable example showing one ``SingleStepDirectLLMAgent`` execution.

The script calls a local llama-cpp server by default, runs one prompt directly
through the model, and prints the resulting ``AgentResult`` payload.
"""

import design_research_agents as dra


def main() -> None:
    """Execute one direct-LLM agent run and print structured output."""
    llm_client = dra.llm.create_default_llm_client()
    agent = dra.agents.SingleStepDirectLLMAgent(llm_client=llm_client)

    result = agent.run(
        prompt="What is two plus two?",
        request_id="example-direct-llm-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
