"""Run one functional router-agent execution."""

import dataclasses
import json

import design_research_agents
import llama_cpp_example_config


def _status_summary_schema() -> dict[str, object]:
    """Return a small JSON schema for repository status summaries."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["justification", "selection"],
        "properties": {
            "justification": {"type": "string"},
            "selection": {
                "type": "string",
                "enum": ["calculator_tool", "text_stats_tool"],
            },
        },
    }


def main() -> None:
    """Execute one router agent run and print structured output."""
    settings = llama_cpp_example_config.configure_example_llama_backend()
    # Use the active backend configured above.
    llm_client = design_research_agents.BaseLLMClient()
    tool_runtime = design_research_agents.BaseToolRuntime()
    agent = design_research_agents.RouterAgent(
        llm_client=llm_client, tool_runtime=tool_runtime, model=settings.api_model
    )

    # Provide explicit alternatives and let RouterAgent choose one tool route.
    result = agent.run(
        input={
            "prompt": "Select which tool to provide a short status summary for this repository.",
            "alternatives": [
                {
                    "tool_name": "calculator_tool",
                    "description": "Use for arithmetic expressions.",
                    "keywords": ["math", "compute"],
                },
                {
                    "tool_name": "text_stats_tool",
                    "description": "Use for text summaries and content analysis.",
                    "keywords": ["summary", "text", "analysis"],
                },
            ],
            "response_schema": _status_summary_schema(),
        },
        context={"request_id": "example-router-agent-001"},
    )

    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
