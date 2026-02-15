"""Runnable streaming example showing one ``MultiStepAgent`` lifecycle."""

from _streaming_support import SequenceResponseLLMClient, print_stream_event

import design_research_agents


def main() -> None:
    """Execute one multi-step streaming run and print events."""
    llm_client = SequenceResponseLLMClient(
        response_texts=[
            '{"continue": true, "reason": "Run one action step."}',
            'result = call_tool("calculator_tool", {"expression": "6 * 7"})\n'
            'final_output = {"result": result["result"], "summary": "Computed in one step."}',
        ]
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.MultiStepAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        model="example-multi-step-model",
        max_steps=1,
        normalize_generated_code_per_step=True,
    )

    for event in agent.run_stream(
        input="Compute 6 * 7, then provide a short summary.",
        request_id="example-multi-step-agent-stream-001",
    ):
        print_stream_event(event)


if __name__ == "__main__":
    main()
