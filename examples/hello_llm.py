"""Minimal runnable example that uses the llama-cpp backend."""

from __future__ import annotations

from design_research_agents import complete, configure_llama_cpp_server

if __name__ == "__main__":
    # Configure llama-cpp
    configure_llama_cpp_server(
        model="tinyllama.Q4_K_M.gguf",
        hf_model_repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    )

    # Query and capture the response.
    response = complete("Say hello", backend="llama-cpp-server")
    print(response)
