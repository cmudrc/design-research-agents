"""Runnable example for ``PlannerExecutorPattern`` workflow orchestration."""

from design_research_agents import (
    LlamaCppServerLLMClient,
    PlannerExecutorPattern,
    Toolbox,
)


def main() -> None:
    """Run planner + executor orchestration with configurable dependencies."""
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
