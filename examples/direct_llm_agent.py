"""Runnable example showing one ``DirectLLMAgent`` execution.

The script configures a local backend, runs one prompt directly through the
model, and prints the resulting ``AgentResult`` payload.
"""

import dataclasses
import json

import design_research_agents
import llama_cpp_example_config


def main() -> None:
    """Execute one direct-LLM agent run and print structured output."""
    llm_client = llama_cpp_example_config.create_example_llm_client()
    agent = design_research_agents.DirectLLMAgent(llm_client=llm_client)

    result = agent.run(
        input={
            "prompt": "What is two plus two?",
        },
        context={"request_id": "example-direct-llm-agent-001"},
    )

    print(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
