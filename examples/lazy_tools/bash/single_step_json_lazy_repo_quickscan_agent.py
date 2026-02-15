"""Runnable example using one ``SingleStepJsonToolCallingAgent`` lazy tool call.

This script enables lazy tool discovery for the Bash lazy-tool example and
asks the model to execute ``lazy::repo_quickscan`` in one step.
"""

from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, UnifiedToolRuntime
from design_research_agents.agent import SingleStepJsonToolCallingAgent


def main() -> None:
    """Run one single-step agent call against ``lazy::repo_quickscan``."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]

    llm_client = LlamaCppServerLLMClient()
    tool_runtime = UnifiedToolRuntime.lazy(
        search_paths=(str(script_dir),),
        workspace_root=str(repo_root),
        enable_core_tools=False,
    )
    agent = SingleStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    result = agent.run(
        prompt="Call lazy::repo_quickscan with include_hidden=false.",
        request_id="example-lazy-repo-quickscan-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
