"""Runnable example showing one ``SingleStepCodeToolCallingAgent`` execution.

The script generates one action program, executes it in the sandbox, and prints
the resulting structured output.
"""

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepCodeToolCallingAgent


def main() -> None:
    """Execute one single-step code-agent run and print ``AgentResult`` data.

    Demonstrates generated-code execution with default sandbox constraints.
    """
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = SingleStepCodeToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        normalize_generated_code=True,
    )

    result = agent.run(
        prompt=(
            "Create a tiny tool inventory CSV in artifacts, describe it, and search "
            "the tools package for Toolbox."
        ),
        request_id="example-single-step-code-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
