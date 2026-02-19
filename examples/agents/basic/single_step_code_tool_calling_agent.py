"""Runnable example showing one ``SingleStepCodeToolCallingAgent`` execution.

The script generates one action program, executes it in the sandbox, and prints
the resulting structured output.
"""

import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepCodeToolCallingAgent


def main() -> None:
    """Execute one single-step code-agent run and print ``ExecutionResult`` data.

    Demonstrates generated-code execution with default sandbox constraints.
    """
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        agent = SingleStepCodeToolCallingAgent(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            normalize_generated_code=True,
            default_tools=({"tool_name": "calculator"},),
        )
        result = agent.run(
            prompt=(
                "No imports. Use call_tool only. Call calculator for 12 * (4 + 1) and "
                "set final_output to a dict that includes the numeric result."
            ),
            request_id="example-single-step-code-agent-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "selected_tool_count": len(result.tool_results),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
