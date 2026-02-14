"""Runnable example showing one ``MultiStepAgent`` execution lifecycle.

The script demonstrates iterative continuation/step execution over a short
multi-step task and prints the final structured result payload.
"""

import dataclasses
import json

import design_research_agents
import llama_cpp_example_config


def main() -> None:
    """Execute one multi-step run and print the resulting ``AgentResult``.

    Demonstrates iterative planning/execution behavior with a bounded step count.
    """
    llm_client = llama_cpp_example_config.create_example_llm_client()
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.MultiStepAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    result = agent.run(
        input={
            "prompt": "Compute 6 * 7, then produce text stats for a one-line summary.",
        },
        context={"request_id": "example-multi-step-agent-001"},
    )

    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
