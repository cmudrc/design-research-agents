"""Runnable example showing one ``MultiStepJsonToolCallingAgent`` execution lifecycle.

The script demonstrates iterative continuation/step execution over a short
multi-step task and prints the final structured result payload.
"""

import json
from collections.abc import Mapping

from design_research_agents import CallableTool, LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import MultiStepJsonToolCallingAgent


def _repo_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    """Return deterministic metrics for one multi-step tool-calling turn.

    Args:
        payload: Optional tool input mapping.

    Returns:
        Deterministic metric payload.
    """
    expression = str(payload.get("expression", "12 * (4 + 1)"))
    result = 60 if expression == "12 * (4 + 1)" else 20
    return {"expression": expression, "result": result}


def main() -> None:
    """Execute one multi-step JSON run and print the resulting ``ExecutionResult``."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="repo.metrics",
                description="Return deterministic repository metrics for demo prompts.",
                handler=_repo_metrics,
            ),
        ),
    )
    try:
        agent = MultiStepJsonToolCallingAgent(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=1,
        )
        result = agent.run(
            prompt="Call repo.metrics with expression '12 * (4 + 1)' and return the result.",
            request_id="example-multi-step-json-tool-calling-agent-001",
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
