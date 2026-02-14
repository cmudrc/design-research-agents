"""Runnable example showing one ``SingleStepCodeAgent`` execution.

The script generates one action program, executes it in the sandbox, and prints
the resulting structured output.
"""

import dataclasses
import json

import design_research_agents


def main() -> None:
    """Execute one single-step code-agent run and print ``AgentResult`` data.

    Demonstrates generated-code execution with default sandbox constraints.
    """
    llm_client = design_research_agents.create_default_llm_client()
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.SingleStepCodeAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        normalize_generated_code=True,
    )

    result = agent.run(
        input="Calculate 12 * (4 + 1), then summarize the numeric result as text stats.",
        request_id="example-single-step-code-agent-001",
    )

    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
