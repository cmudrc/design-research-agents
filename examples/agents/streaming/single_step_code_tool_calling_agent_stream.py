"""Runnable streaming example showing one ``SingleStepCodeToolCallingAgent`` execution."""

import dataclasses
import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepCodeToolCallingAgent
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
    """Execute one single-step code-agent streaming run and print events."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = SingleStepCodeToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        normalize_generated_code=True,
    )

    for event in agent.run_stream(
        prompt="Summarize top-level repo size and number of Toolbox references in src.",
        request_id="example-single-step-code-agent-stream-001",
    ):
        _print_stream_event(event)


if __name__ == "__main__":
    main()
