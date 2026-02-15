"""Runnable example showing one ``ToolCallingAgent`` execution flow.

The script uses a deterministic in-process LLM stub and runs a single arithmetic request
through model-selected tool invocation.
"""

import json

from _basic_support import RecordingSequenceLLMClient

import design_research_agents


def main() -> None:
    """Execute one tool-calling run and print structured ``AgentResult`` output.

    This entrypoint shows model-guided tool selection in a single step.
    """
    llm_client = RecordingSequenceLLMClient(
        response_texts=[
            json.dumps(
                {
                    "tool_name": "calculator_tool",
                    "tool_input": {"expression": "12 * (4 + 1)"},
                }
            )
        ]
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.ToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    result = agent.run(
        input="Calculate this expression and return the numeric result: 12 * (4 + 1)",
        request_id="example-tool-calling-agent-001",
    )

    tool_call_metadata = result.metadata.get("tool_call")
    if isinstance(tool_call_metadata, dict) and tool_call_metadata.get("source") != "model":
        raise ValueError("Unexpected fallback tool selection in ToolCallingAgent example.")
    llm_client.assert_exhausted()
    print(result)


if __name__ == "__main__":
    main()
