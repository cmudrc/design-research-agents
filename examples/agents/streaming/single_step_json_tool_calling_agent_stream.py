"""Runnable streaming example showing one ``SingleStepJsonToolCallingAgent`` execution flow."""

import dataclasses
import json

import design_research_agents as dra


def _print_stream_event(event: dra.contracts.agent.AgentStreamEvent) -> None:
    if event.kind == "delta":
        print(f"delta: {event.delta_text or ''}")
        return
    if event.result is None:
        print("completed: null")
        return
    print("completed:")
    print(json.dumps(dataclasses.asdict(event.result), indent=2, sort_keys=True))


def main() -> None:
    """Execute one tool-calling streaming run and print events."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    agent = dra.agents.SingleStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    for event in agent.run_stream(
        prompt="Calculate this expression and return only the numeric result: 12 * (4 + 1)",
        request_id="example-tool-calling-agent-stream-001",
    ):
        _print_stream_event(event)


if __name__ == "__main__":
    main()
