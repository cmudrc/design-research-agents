"""Runnable example using one ``SingleStepJsonToolCallingAgent`` lazy tool call.

This script enables lazy tool discovery for the Python lazy-tool example and
asks the model to execute ``lazy::rubric_score`` in one step.
"""

from pathlib import Path

import design_research_agents as dra


def main() -> None:
    """Run one single-step agent call against ``lazy::rubric_score``."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]

    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime(
        config=dra.tools.ToolRuntimeConfig(
            core_tools=dra.tools.CoreToolsConfig(
                enabled=False,
                workspace_root=str(repo_root),
            ),
            lazy_tools=dra.tools.LazyToolsConfig(
                enabled=True,
                search_paths=(str(script_dir),),
            ),
        )
    )
    agent = dra.agents.SingleStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    result = agent.run(
        prompt=(
            "Call lazy::rubric_score with text 'Agents can quickly score this sample summary.' "
            "and max_score 12."
        ),
        request_id="example-lazy-rubric-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
