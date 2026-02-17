"""Runnable streaming example showing one ``SingleStepToolRouterAgent`` flow."""

import dataclasses
import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepToolRouterAgent
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
    """Execute one streaming tool-router run and print events."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = SingleStepToolRouterAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    for event in agent.run_stream(
        prompt="Calculate this expression and return only the numeric result: 12 * (4 + 1)",
        request_id="example-tool-router-agent-stream-001",
    ):
        _print_stream_event(event)


if __name__ == "__main__":
    main()
