"""Runnable example for ``ReflexionPattern`` workflow orchestration."""

import json

from design_research_agents import (
    LlamaCppServerLLMClient,
    ReflexionPattern,
    Toolbox,
)


def main() -> None:
    """Run propose/critique refinement orchestration with configurable dependencies."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = ReflexionPattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
        )
        result = workflow.run(
            prompt="Write a short design summary for this repository.",
            request_id="example-propose-critic-workflow-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "approved": output.get("approved"),
        "critique_iterations": output.get("critique_iterations"),
        "proposal": output.get("proposal"),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
