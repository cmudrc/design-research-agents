"""Runnable example showing one ``SingleStepCodeToolCallingAgent`` execution.

The script generates one action program, executes it in the sandbox, and prints
the resulting structured output.
"""

import design_research_agents as dra


def main() -> None:
    """Execute one single-step code-agent run and print ``AgentResult`` data.

    Demonstrates generated-code execution with default sandbox constraints.
    """
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    agent = dra.agents.SingleStepCodeToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        normalize_generated_code=True,
    )

    result = agent.run(
        prompt=(
            "Create a tiny tool inventory CSV in artifacts, describe it, and search "
            "the tools package for UnifiedToolRuntime."
        ),
        request_id="example-single-step-code-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
