"""Runnable streaming example showing one ``MultiStepDirectLLMAgent`` lifecycle."""

import dataclasses
import json

from design_research_agents import LlamaCppServerLLMClient
from design_research_agents.agent import MultiStepDirectLLMAgent
from design_research_agents.contracts.agent import AgentStreamEvent


def _print_stream_event(event: AgentStreamEvent) -> None:
    if event.kind == "delta":
        print(f"delta: {event.delta_text or ''}")
        return
    if event.result is None:
        print("completed: null")
        return
    print("completed:")
    print(json.dumps(dataclasses.asdict(event.result), indent=2, sort_keys=True))


def main() -> None:
    """Execute one multi-step direct-LLM streaming run and print events."""
    llm_client = LlamaCppServerLLMClient()
    agent = MultiStepDirectLLMAgent(
        llm_client=llm_client,
        max_steps=3,
    )

    for event in agent.run_stream(
        prompt="Draft and then finalize a concise answer to what 6 * 7 equals.",
        request_id="example-multi-step-direct-llm-agent-stream-001",
    ):
        _print_stream_event(event)


if __name__ == "__main__":
    main()
