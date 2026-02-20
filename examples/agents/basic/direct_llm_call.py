"""Runnable example showing one ``DirectLLMCall`` execution.

The script calls a local llama-cpp server by default, runs one prompt directly
through the model, and prints the resulting ``ExecutionResult`` payload.
"""

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient


def main() -> None:
    """Execute one direct LLM call and print structured output."""
    llm_client = LlamaCppServerLLMClient()
    agent = DirectLLMCall(llm_client=llm_client)

    result = agent.run(
        prompt="What is two plus two?",
        request_id="example-direct-llm-call-001",
    )

    print(result)


if __name__ == "__main__":
    main()
