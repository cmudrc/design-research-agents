"""Runnable streaming example showing one ``DirectLLMAgent`` execution."""

from _streaming_support import StaticResponseLLMClient, print_stream_event

import design_research_agents


def main() -> None:
    """Execute one direct-LLM streaming run and print events."""
    llm_client = StaticResponseLLMClient(response_text="The answer is 4.")
    agent = design_research_agents.DirectLLMAgent(
        llm_client=llm_client,
        model="example-direct-model",
    )

    for event in agent.run_stream(
        prompt="What is two plus two?",
        request_id="example-direct-llm-agent-stream-001",
    ):
        print_stream_event(event)


if __name__ == "__main__":
    main()
