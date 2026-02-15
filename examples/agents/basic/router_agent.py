"""Runnable example showing one ``RouterAgent`` execution end-to-end.

The script uses a deterministic in-process LLM stub, builds runtime/tool dependencies, and
executes runtime-driven route selection with built-in default schemas.
"""

import json

from _basic_support import RecordingSequenceLLMClient

import design_research_agents


def main() -> None:
    """Execute one router-agent run and print structured ``AgentResult`` output.

    Demonstrates route selection plus downstream tool invocation in one call.
    """
    llm_client = RecordingSequenceLLMClient(
        response_texts=[
            json.dumps({"selection": "text.word_count", "reason": "Analyze text content."}),
        ]
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.RouterAgent(llm_client=llm_client, tool_runtime=tool_runtime)

    # RouterAgent will derive available routes from ToolRuntime.list_tools().
    result = agent.run(
        prompt="Select which tool to provide a short status summary for this repository.",
        request_id="example-router-agent-001",
    )

    routing_metadata = result.metadata.get("routing")
    if isinstance(routing_metadata, dict) and routing_metadata.get("source") != "model":
        raise ValueError("Unexpected non-model routing source in RouterAgent example.")
    llm_client.assert_exhausted()
    print(result)


if __name__ == "__main__":
    main()
