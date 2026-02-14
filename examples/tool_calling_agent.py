"""Run one functional tool-calling agent execution."""

import dataclasses
import json

import design_research_agents
import llama_cpp_example_config


def main() -> None:
    """Execute one tool-calling agent run and print structured output."""
    settings = llama_cpp_example_config.configure_example_llama_backend()
    llm_client = design_research_agents.BaseLLMClient()
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.ToolCallingAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        model=settings.api_model,
    )

    result = agent.run(
        input={
            "prompt": "Calculate this expression and return the numeric result: 12 * (4 + 1)",
            "tools": [
                {
                    "tool_name": "calculator_tool",
                    "description": "Evaluate arithmetic expressions.",
                },
                {
                    "tool_name": "text_stats_tool",
                    "description": "Compute text statistics like word counts.",
                },
            ],
        },
        context={"request_id": "example-tool-calling-agent-001"},
    )

    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
