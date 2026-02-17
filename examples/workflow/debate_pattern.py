"""Runnable example for the reusable ``DebatePattern`` orchestration chunk."""

from design_research_agents import DebatePattern, LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import RuntimeControls


def main() -> None:
    """Run ``DebatePattern`` with one debate round and a final judge verdict."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    workflow = DebatePattern(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        controls=RuntimeControls(max_iterations=1),
    )
    result = workflow.run(
        prompt="Should a research team prioritize local models over hosted APIs?",
        request_id="example-debate-pattern-001",
    )
    print(result)


if __name__ == "__main__":
    main()
