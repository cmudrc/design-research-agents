"""Runnable llama-cpp backend example."""

import design_research_agents
import llama_cpp_example_config

PROMPT = "What is two plus two?"

if __name__ == "__main__":
    llama_cpp_example_config.configure_example_llama_backend()
    # Hardcoded backend settings keep this runnable without environment variables.
    response = design_research_agents.complete(PROMPT)
    print(response)
