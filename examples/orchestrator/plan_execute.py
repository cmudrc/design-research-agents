"""Runnable example for the reusable ``plan_execute`` orchestration chunk."""

import design_research_agents as dra
from design_research_agents.orchestrator.implementations.plan_execute import plan_and_execute


def main() -> None:
    """Run ``plan_execute`` orchestration with configurable runtime dependencies."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    orchestrator = plan_and_execute(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )
    result = orchestrator.run(
        prompt="Create and analyze a tiny runtime tools inventory.",
        request_id="example-plan-execute-orchestrator-001",
    )
    print(result)


if __name__ == "__main__":
    main()
