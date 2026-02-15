"""Runnable example for the reusable ``propose_critic`` orchestration chunk."""

import design_research_agents as dra


def main() -> None:
    """Run ``propose_critic`` orchestration with configurable dependencies."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    workflow = dra.workflows.ProposeAndCritiqueWorkflow(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )
    result = workflow.run(
        prompt="Write a short design summary for this repository.",
        request_id="example-propose-critic-workflow-001",
    )
    print(result)


if __name__ == "__main__":
    main()
