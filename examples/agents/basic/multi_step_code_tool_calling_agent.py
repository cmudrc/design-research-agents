"""Runnable example showing one ``MultiStepCodeToolCallingAgent`` execution lifecycle.

The script demonstrates iterative continuation/step execution over a short
multi-step task and prints the final structured result payload.
"""

import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import MultiStepCodeToolCallingAgent


def main() -> None:
    """Execute one multi-step run and print the resulting ``ExecutionResult``.

    Demonstrates iterative planning/execution behavior with a bounded step count.
    """
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        agent = MultiStepCodeToolCallingAgent(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            normalize_generated_code_per_step=True,
            default_tools_per_step=({"tool_name": "calculator"},),
        )
        result = agent.run(
            prompt=(
                "No imports. Use call_tool only. In one or more steps, compute 12 * (4 + 1) "
                "and then compute 60 / 3. Return a compact final_output dict with both values."
            ),
            request_id="example-multi-step-agent-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "steps_executed": output.get("steps_executed"),
        "step_outputs_count": len(output.get("step_outputs", []))
        if isinstance(output.get("step_outputs"), list)
        else 0,
        "tool_results_count": len(result.tool_results),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
