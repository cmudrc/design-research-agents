"""Runnable example showing one ``MultiStepAgent`` execution lifecycle.

The script demonstrates iterative continuation/step execution over a short
multi-step task and prints the final structured result payload.
"""

import json

from _basic_support import RecordingSequenceLLMClient

import design_research_agents


def main() -> None:
    """Execute one multi-step run and print the resulting ``AgentResult``.

    Demonstrates iterative planning/execution behavior with a bounded step count.
    """
    llm_client = RecordingSequenceLLMClient(
        response_texts=[
            json.dumps({"continue": True, "thought": "Use calculator first."}),
            "\n".join(
                [
                    'calc = call_tool("calculator", {"expression": "6 * 7"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
            json.dumps({"continue": False, "thought": "Task complete."}),
        ]
    )
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.MultiStepAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
        normalize_generated_code_per_step=True,
    )

    result = agent.run(
        prompt="Compute 6 * 7.",
        request_id="example-multi-step-agent-001",
    )

    continuation_history = result.output.get("continuation_history", [])
    if any(
        isinstance(entry, dict) and entry.get("source") == "fallback"
        for entry in continuation_history
    ):
        raise ValueError("Unexpected continuation fallback in MultiStepAgent example.")
    llm_client.assert_exhausted()
    print(result)


if __name__ == "__main__":
    main()
