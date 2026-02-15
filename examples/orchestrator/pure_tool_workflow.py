"""Runnable example for the reusable pure-tool workflow orchestration chunk."""

import design_research_agents as dra
from design_research_agents.orchestrator.implementations.pure_tool_workflow import (
    pure_tool_workflow,
)


def main() -> None:
    """Run the configured pure-tool workflow and print serialized output."""
    orchestrator = pure_tool_workflow(
        tool_runtime=dra.tools.UnifiedToolRuntime(),
    )
    result = orchestrator.run(execution_mode="sequential")
    print(result.asdict())


if __name__ == "__main__":
    main()
