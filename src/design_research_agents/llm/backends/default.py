"""Default local llama-cpp configuration helpers for this package.

This module centralizes shared llama-cpp settings so examples and local
experiments use one consistent default configuration under ``llm.backends``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from design_research_agents.llm.base_client import BaseLLMClient


@dataclasses.dataclass(frozen=True, slots=True)
class DefaultLlamaCppSettings:
    """Hardcoded default llama-cpp settings used by local runs.

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


DEFAULT_LLAMA_CPP_SETTINGS = DefaultLlamaCppSettings()


def configure_default_llama_cpp_backend() -> DefaultLlamaCppSettings:
    """Configure local llama-cpp backend settings and return what was applied.

    Returns:
        Default llama-cpp settings applied to the backend configuration.
    """
    settings = DEFAULT_LLAMA_CPP_SETTINGS
    # Import lazily to avoid cyclic imports with runtime configuration module.
    from design_research_agents.llm import configure_llama_cpp_server

    configure_llama_cpp_server(
        model=settings.model,
        hf_model_repo_id=settings.hf_model_repo_id,
        api_model=settings.api_model,
        host=settings.host,
        port=settings.port,
    )
    return settings


def create_default_llm_client() -> BaseLLMClient:
    """Create an LLM client using package default local llama-cpp settings.

    Returns:
        Base client configured for the default local llama-cpp backend.
    """
    settings = DEFAULT_LLAMA_CPP_SETTINGS
    # Import lazily to avoid cyclic imports with runtime configuration module.
    from design_research_agents.llm.base_client import BaseLLMClient

    return BaseLLMClient.from_llama_cpp_server(
        model=settings.model,
        hf_model_repo_id=settings.hf_model_repo_id,
        api_model=settings.api_model,
        host=settings.host,
        port=settings.port,
    )
