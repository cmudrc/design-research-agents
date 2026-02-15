"""Runnable example for the reusable ``plan_execute`` orchestration chunk."""

import design_research_agents as dra


def main() -> None:
    """Run ``plan_execute`` orchestration with configurable runtime dependencies."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    workflow = dra.workflows.PlanExecuteWorkflow(
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
