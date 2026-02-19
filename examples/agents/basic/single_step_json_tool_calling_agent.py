"""Runnable example showing one ``SingleStepJsonToolCallingAgent`` execution flow.

The script calls a local llama-cpp server by default and runs a single arithmetic request
through model-selected tool invocation.
"""

import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepJsonToolCallingAgent


def main() -> None:
    """Execute one tool-calling run and print structured ``ExecutionResult`` output.

    This entrypoint shows model-guided tool selection in a single step.
    """
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        agent = SingleStepJsonToolCallingAgent(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
        )
        result = agent.run(
            prompt="Calculate this expression and return the numeric result: 12 * (4 + 1)",
            request_id="example-tool-calling-agent-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "selected_tool": output.get("tool_name"),
        "tool_input": output.get("tool_input"),
        "tool_output": output.get("tool_output"),
        "tool_results_count": len(result.tool_results),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
