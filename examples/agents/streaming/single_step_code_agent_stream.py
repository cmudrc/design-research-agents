"""Runnable streaming example showing one ``SingleStepCodeAgent`` execution."""

from _streaming_support import StaticResponseLLMClient, print_stream_event

import design_research_agents


def main() -> None:
    """Execute one single-step code-agent streaming run and print events."""
    llm_client = StaticResponseLLMClient(
        response_text=(
            'result = call_tool("calculator", {"expression": "12 * (4 + 1)"})\n'
            'final_output = {"expression": result["expression"], "result": result["result"]}'
        )
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.SingleStepCodeAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        model="example-single-step-code-model",
        normalize_generated_code=True,
    )

    for event in agent.run_stream(
        prompt="Calculate 12 * (4 + 1) and return the result.",
        request_id="example-single-step-code-agent-stream-001",
    ):
        print_stream_event(event)


if __name__ == "__main__":
    main()
