"""Runnable example for ``PlannerExecutorPattern`` workflow orchestration."""

import json
from collections.abc import Mapping
from pathlib import Path

from design_research_agents import (
    CallableTool,
    LlamaCppServerLLMClient,
    PlannerExecutorPattern,
    Toolbox,
)


def _readme_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    """Return basic README metrics for plan/execute demos.

    Args:
        payload: Optional tool input mapping.

    Returns:
        README metric payload.
    """
    del payload
    readme_path = Path("README.md")
    readme_text = readme_path.read_text(encoding="utf-8")
    lines = readme_text.splitlines()
    first_heading = next(
        (line.lstrip("#").strip() for line in lines if line.startswith("#")),
        "",
    )
    return {
        "path": str(readme_path),
        "line_count": len(lines),
        "first_heading": first_heading,
    }


def main() -> None:
    """Run planner + executor orchestration with configurable dependencies."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="repo.readme_metrics",
                description="Return README line-count and first heading.",
                handler=_readme_metrics,
            ),
        ),
    )
    try:
        workflow = PlannerExecutorPattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_iterations=1,
        )
        result = workflow.run(
            prompt=(
                "Use repo.readme_metrics to gather README stats and return a concise "
                "execution summary."
            ),
            request_id="example-plan-execute-workflow-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    plan_payload = output.get("plan")
    plan_steps = plan_payload.get("steps") if isinstance(plan_payload, dict) else None
    payload = {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "steps_executed": output.get("steps_executed"),
        "plan_step_count": len(plan_steps) if isinstance(plan_steps, list) else 0,
        "final_output": output.get("final_output"),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
