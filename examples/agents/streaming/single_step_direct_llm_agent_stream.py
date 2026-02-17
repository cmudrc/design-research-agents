"""Runnable streaming example showing one ``SingleStepDirectLLMAgent`` execution."""

import dataclasses
import json

from design_research_agents import LlamaCppServerLLMClient
from design_research_agents.agent import SingleStepDirectLLMAgent
from design_research_agents.contracts.agent import AgentStreamEvent


def _print_stream_event(event: AgentStreamEvent) -> None:
    """Run print stream event.

    Args:
        event: Parameter value.
    """
    if event.kind == "delta":
        print(f"delta: {event.delta_text or ''}")
        return
    if event.result is None:
        print("completed: null")
        return
    print("completed:")
    print(json.dumps(dataclasses.asdict(event.result), indent=2, sort_keys=True))


def main() -> None:
    """Execute one direct-LLM streaming run and print events."""
    llm_client = LlamaCppServerLLMClient()
    agent = SingleStepDirectLLMAgent(
        llm_client=llm_client,
    )

    for event in agent.run_stream(
        prompt="What is two plus two?",
        request_id="example-direct-llm-agent-stream-001",
    ):
        _print_stream_event(event)


if __name__ == "__main__":
    main()
