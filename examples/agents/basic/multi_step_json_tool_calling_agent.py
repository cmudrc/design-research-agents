"""Runnable example showing one ``MultiStepJsonToolCallingAgent`` execution lifecycle.

The script demonstrates iterative continuation/step execution over a short
multi-step task and prints the final structured result payload.
"""

from design_research_agents import LlamaCppServerLLMClient, UnifiedToolRuntime
from design_research_agents.agent import MultiStepJsonToolCallingAgent


def main() -> None:
    """Execute one multi-step JSON run and print the resulting ``AgentResult``."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = UnifiedToolRuntime()
    agent = MultiStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    result = agent.run(
        prompt="Read README and return compact repository text metrics.",
        request_id="example-multi-step-json-tool-calling-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
