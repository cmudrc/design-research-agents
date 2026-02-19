"""Runnable example showing one ``SingleStepToolRouterAgent`` execution end-to-end."""

import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepToolRouterAgent


def main() -> None:
    """Execute one tool-router run and print structured ``ExecutionResult`` output."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        agent = SingleStepToolRouterAgent(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            allowed_routes=("calculator",),
        )
        result = agent.run(
            prompt="Use calculator to evaluate 12 * (4 + 1).",
            request_id="example-tool-router-agent-001",
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
