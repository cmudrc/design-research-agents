"""Runnable example showing one ``MultiStepCodeToolCallingAgent`` execution lifecycle.

The script demonstrates iterative continuation/step execution over a short
multi-step task and prints the final structured result payload.
"""

import design_research_agents as dra


def main() -> None:
    """Execute one multi-step run and print the resulting ``AgentResult``.

    Demonstrates iterative planning/execution behavior with a bounded step count.
    """
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    agent = dra.agents.MultiStepCodeToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
        normalize_generated_code_per_step=True,
    )

    result = agent.run(
        prompt="Analyze README size and show a tiny diff that mentions new tool sources.",
        request_id="example-multi-step-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
