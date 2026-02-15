"""Runnable streaming example showing one ``ToolCallingAgent`` execution flow."""

from _streaming_support import StaticResponseLLMClient, print_stream_event

import design_research_agents


def main() -> None:
    """Execute one tool-calling streaming run and print events."""
    llm_client = StaticResponseLLMClient(
        response_text=(
            '{"tool_name":"calculator_tool","tool_input":{"expression":"12 * (4 + 1)"},'
            '"reason":"Arithmetic request."}'
        )
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.ToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        model="example-tool-calling-model",
    )

    for event in agent.run_stream(
        input="Calculate this expression and return only the numeric result: 12 * (4 + 1)",
        request_id="example-tool-calling-agent-stream-001",
    ):
        print_stream_event(event)


if __name__ == "__main__":
    main()
