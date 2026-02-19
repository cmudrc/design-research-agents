"""Runnable example showing one ``MultiStepToolRouterAgent`` execution lifecycle."""

import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import MultiStepToolRouterAgent


def main() -> None:
    """Execute one multi-step tool-router run and print the resulting result."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        agent = MultiStepToolRouterAgent(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
        )
        result = agent.run(
            prompt="Compute 12 * (4 + 1), then stop with a final structured output.",
            request_id="example-multi-step-tool-router-agent-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "steps_executed": output.get("steps_executed"),
        "tool_results_count": len(result.tool_results),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
