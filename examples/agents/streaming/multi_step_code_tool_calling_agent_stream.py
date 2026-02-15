"""Runnable streaming example showing one ``MultiStepCodeToolCallingAgent`` lifecycle."""

import dataclasses
import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import MultiStepCodeToolCallingAgent
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
    """Execute one multi-step streaming run and print events."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = MultiStepCodeToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=1,
        normalize_generated_code_per_step=True,
    )

    for event in agent.run_stream(
        prompt="Measure README size and provide a short summary.",
        request_id="example-multi-step-agent-stream-001",
    ):
        _print_stream_event(event)


if __name__ == "__main__":
    main()
