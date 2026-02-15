"""Runnable example for ``AgentRuntime`` in ``plan_execute`` mode."""

from __future__ import annotations

import json

from _runtime_example_support import SequenceResponseLLMClient

import design_research_agents


def main() -> None:
    llm_client = SequenceResponseLLMClient(
        response_texts=[
            json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "compute",
                            "instruction": "Compute 6 * 7.",
                            "success_criteria": "Return numeric result.",
                        }
                    ]
                }
            ),
            "\n".join(
                [
                    'calc = call_tool("calculator_tool", {"expression": "6 * 7"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
        ]
    )

    agent = design_research_agents.AgentRuntime(
        llm_client=llm_client,
        tool_runtime=design_research_agents.BaseToolRuntime(),
        mode="plan_execute",
    )
    result = agent.run("Compute 6 * 7.")
    print(result)


if __name__ == "__main__":
    main()
