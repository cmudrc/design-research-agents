"""Runnable example showing one ``ToolCallingAgent`` execution flow.

The script configures the local backend and runs a single arithmetic request
through model-selected tool invocation.
"""

import dataclasses
import json

import design_research_agents


def main() -> None:
    """Execute one tool-calling run and print structured ``AgentResult`` output.

    This entrypoint shows model-guided tool selection in a single step.
    """
    llm_client = design_research_agents.create_default_llm_client()
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.ToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    result = agent.run(
        input="Calculate this expression and return the numeric result: 12 * (4 + 1)",
        request_id="example-tool-calling-agent-001",
    )

    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
