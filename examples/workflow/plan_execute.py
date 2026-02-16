"""Runnable example for the reusable ``plan_execute`` orchestration chunk."""

from design_research_agents import (
    LlamaCppServerLLMClient,
    PlannerExecutorPattern,
    Toolbox,
)


def main() -> None:
    """Run ``plan_execute`` orchestration with configurable runtime dependencies."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    workflow = PlannerExecutorPattern(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )
    result = workflow.run(
        prompt="Create and analyze a tiny runtime tools inventory.",
        request_id="example-plan-execute-workflow-001",
    )
    print(result)


if __name__ == "__main__":
    main()
