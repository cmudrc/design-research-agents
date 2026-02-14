"""Shared llama-cpp helpers used by local example scripts.

Keeping example backend settings in one module ensures every runnable example
uses the same model/server configuration with minimal boilerplate.
"""

import dataclasses

import design_research_agents


@dataclasses.dataclass(frozen=True, slots=True)
class ExampleLlamaSettings:
    """Hardcoded llama-cpp settings used by local runnable examples.

    Attributes:
        api_model: OpenAI-compatible model alias served by llama-cpp.
        model: GGUF model filename passed to ``llama_cpp.server --model``.
        hf_model_repo_id: Hugging Face repository containing ``model``.
        host: Host used by the local server.
        port: Port used by the local server.
    """

    api_model: str = "qwen2-1.5b-q4"
    model: str = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    hf_model_repo_id: str = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    host: str = "127.0.0.1"
    port: int = 8001


DEFAULT_LLAMA_SETTINGS = ExampleLlamaSettings()


def configure_example_llama_backend() -> ExampleLlamaSettings:
    """Configure the local llama-cpp backend used by examples.

    Returns:
        Hardcoded settings that were applied.
    """
    settings = DEFAULT_LLAMA_SETTINGS
    # Configure the managed local server before creating any client calls.
    design_research_agents.configure_llama_cpp_server(
        model=settings.model,
        hf_model_repo_id=settings.hf_model_repo_id,
        api_model=settings.api_model,
        host=settings.host,
        port=settings.port,
    )
    return settings


def create_example_llm_client() -> design_research_agents.BaseLLMClient:
    """Create an example client while applying local llama-cpp settings."""
    settings = DEFAULT_LLAMA_SETTINGS
    return design_research_agents.BaseLLMClient.from_llama_cpp_server(
        model=settings.model,
        hf_model_repo_id=settings.hf_model_repo_id,
        api_model=settings.api_model,
        host=settings.host,
        port=settings.port,
    )
