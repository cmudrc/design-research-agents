"""Runnable example using one ``SingleStepJsonToolCallingAgent`` script tool call.

This script configures one script tool for the Python example and
asks the model to execute ``script::rubric_score`` in one step.
"""

from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepJsonToolCallingAgent
from design_research_agents.tools.config import ScriptTool


def main() -> None:
    """Run one single-step agent call against ``script::rubric_score``."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[3]

    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox(
        workspace_root=str(repo_root),
        enable_core_tools=False,
        script_tools=(
            ScriptTool(
                name="rubric_score",
                path=str(script_dir / "rubric_score.py"),
                description="Score text against a simple rubric.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "max_score": {"type": "integer"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object"},
                filesystem_write=True,
            ),
        ),
    )
    agent = SingleStepJsonToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
    )

    result = agent.run(
        prompt=(
            "Call script::rubric_score with text 'Agents can quickly score this sample summary.' "
            "and max_score 12."
        ),
        request_id="example-script-rubric-agent-001",
    )

    print(result)


if __name__ == "__main__":
    main()
