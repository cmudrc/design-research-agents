"""Runnable example showing one ``DirectLLMAgent`` execution.

The script uses a deterministic in-process LLM stub, runs one prompt directly
through the model, and prints the resulting ``AgentResult`` payload.
"""

from _basic_support import RecordingSequenceLLMClient

import design_research_agents


def main() -> None:
    """Execute one direct-LLM agent run and print structured output."""
    llm_client = RecordingSequenceLLMClient(response_texts=["4"])
    agent = design_research_agents.DirectLLMAgent(llm_client=llm_client)

    result = agent.run(
        input="What is two plus two?",
        request_id="example-direct-llm-agent-001",
    )

    if llm_client.generate_calls == 0:
        raise ValueError("Expected generate() path for DirectLLMAgent.")
    llm_client.assert_exhausted()
    print(result)


if __name__ == "__main__":
    main()
