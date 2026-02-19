"""Runnable example using one ``SingleStepJsonToolCallingAgent`` script tool call.

This script configures one script tool for the Python example and
asks the model to execute ``script::rubric_score`` in one step.
"""

import json
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

    try:
        result = agent.run(
            prompt=(
                "Call script::rubric_score with text 'Agents can quickly score this sample "
                "summary.' and max_score 12."
            ),
            request_id="example-script-rubric-agent-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "selected_tool": output.get("tool_name"),
        "tool_input": output.get("tool_input"),
        "tool_output": output.get("tool_output"),
        "tool_results_count": len(result.tool_results),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
