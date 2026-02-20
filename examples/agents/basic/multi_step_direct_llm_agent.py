"""Runnable example showing one ``MultiStepAgent(mode="direct")`` lifecycle."""

import json

from design_research_agents import LlamaCppServerLLMClient
from design_research_agents.agent import MultiStepAgent


def main() -> None:
    """Execute one multi-step direct-LLM run and print the resulting result."""
    llm_client = LlamaCppServerLLMClient()
    try:
        agent = MultiStepAgent(
            mode="direct",
            llm_client=llm_client,
            max_steps=3,
        )
        result = agent.run(
            prompt="Draft then finalize a concise answer to: what is 6 * 7?",
            request_id="example-multi-step-direct-llm-agent-001",
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
