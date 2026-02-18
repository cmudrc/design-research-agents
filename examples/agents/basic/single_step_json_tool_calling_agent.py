"""Runnable example showing one ``SingleStepJsonToolCallingAgent`` execution flow.

The script calls a local llama-cpp server by default and runs a single arithmetic request
through model-selected tool invocation.
"""

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepJsonToolCallingAgent


def main() -> None:
    """Execute one tool-calling run and print structured ``ExecutionResult`` output.

    This entrypoint shows model-guided tool selection in a single step.
    """
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    agent = SingleStepJsonToolCallingAgent(
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
