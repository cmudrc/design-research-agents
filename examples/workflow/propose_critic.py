"""Runnable example for ``ReflexionPattern`` workflow orchestration."""

from design_research_agents import (
    LlamaCppServerLLMClient,
    ReflexionPattern,
    Toolbox,
)


def main() -> None:
    """Run propose/critique refinement orchestration with configurable dependencies."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    workflow = ReflexionPattern(
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
