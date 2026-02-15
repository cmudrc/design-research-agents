"""Runnable example showing one ``SingleStepJsonToolCallingAgent`` execution flow.

The script calls a local llama-cpp server by default and runs a single arithmetic request
through model-selected tool invocation.
"""

import design_research_agents as dra


def main() -> None:
    """Execute one tool-calling run and print structured ``AgentResult`` output.

    This entrypoint shows model-guided tool selection in a single step.
    """
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    agent = dra.agents.SingleStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    result = agent.run(
        prompt="Calculate this expression and return the numeric result: 12 * (4 + 1)",
        request_id="example-tool-calling-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
