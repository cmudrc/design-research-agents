"""Runnable streaming example showing one ``MultiStepToolRouterAgent`` lifecycle."""

import dataclasses
import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import MultiStepToolRouterAgent
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
    """Execute one multi-step tool-router streaming run and print events."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = MultiStepToolRouterAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    for event in agent.run_stream(
        prompt="Compute 12 * (4 + 1), then stop with final_output.",
        request_id="example-multi-step-tool-router-agent-stream-001",
    ):
        _print_stream_event(event)


if __name__ == "__main__":
    main()
