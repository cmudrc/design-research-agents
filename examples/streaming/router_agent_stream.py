"""Runnable streaming example showing one ``RouterAgent`` execution flow."""

from _streaming_support import StaticResponseLLMClient, print_stream_event

import design_research_agents


def main() -> None:
    """Execute one router-agent streaming run and print events."""
    llm_client = StaticResponseLLMClient(
        response_text='{"selection":"calculator_tool","reason":"Arithmetic request."}'
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.RouterAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        model="example-router-model",
    )

    for event in agent.run_stream(
        input="Calculate this expression and return only the numeric result: 12 * (4 + 1)",
        request_id="example-router-agent-stream-001",
    ):
        print_stream_event(event)


if __name__ == "__main__":
    main()
