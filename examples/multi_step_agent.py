"""Run one functional multi-step agent execution."""

import dataclasses
import json

import design_research_agents
import llama_cpp_example_config


def main() -> None:
    """Execute one multi-step run and print structured output."""
    settings = llama_cpp_example_config.configure_example_llama_backend()
    llm_client = design_research_agents.BaseLLMClient()
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.MultiStepAgent(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        model=settings.api_model,
        max_steps=3,
    )

    result = agent.run(
        input={
            "prompt": "Compute 6 * 7, then produce text stats for a one-line summary.",
            "tools": [
                {
                    "tool_name": "calculator_tool",
                    "description": "Evaluate arithmetic expressions.",
                },
                {
                    "tool_name": "text_stats_tool",
                    "description": "Compute text statistics.",
                },
            ],
        },
        context={"request_id": "example-multi-step-agent-001"},
    )

    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
