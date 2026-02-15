"""Runnable example for the reusable ``propose_critic`` orchestration chunk."""

import design_research_agents as dra


def main() -> None:
    """Run ``propose_critic`` orchestration with configurable dependencies."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    orchestrator = dra.workflows.ProposeAndCritiqueOrchestrator(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )
    result = orchestrator.run(
        prompt="Write a short design summary for this repository.",
        request_id="example-propose-critic-orchestrator-001",
    )
    print(result)


if __name__ == "__main__":
    main()
