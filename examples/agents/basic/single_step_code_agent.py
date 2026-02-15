"""Runnable example showing one ``SingleStepCodeAgent`` execution.

The script generates one action program, executes it in the sandbox, and prints
the resulting structured output.
"""

from _basic_support import RecordingSequenceLLMClient

import design_research_agents


def main() -> None:
    """Execute one single-step code-agent run and print ``AgentResult`` data.

    Demonstrates generated-code execution with default sandbox constraints.
    """
    llm_client = RecordingSequenceLLMClient(
        response_texts=[
            "\n".join(
                [
                    'calc = call_tool("calculator_tool", {"expression": "12 * (4 + 1)"})',
                    'final_output = {"result": calc["result"]}',
                ]
            )
        ]
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.SingleStepCodeAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        normalize_generated_code=True,
    )

    result = agent.run(
        input="Calculate 12 * (4 + 1), then summarize the numeric result as text stats.",
        request_id="example-single-step-code-agent-001",
    )

    llm_client.assert_exhausted()
    print(result)


if __name__ == "__main__":
    main()
