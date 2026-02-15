"""Default local llama-cpp configuration helpers for this package.

This module centralizes shared llama-cpp settings so examples and local
experiments use one consistent default configuration under ``llm.backends``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from design_research_agents.llm.backends.factory import backend_config_hash
from design_research_agents.llm.backends.providers.llama_cpp import LlamaCppBackend
from design_research_agents.llm.backends.providers.llama_cpp_server import (
    create_backend as create_llama_cpp_server,
)
from design_research_agents.llm.config import LlamaCppConfig
from design_research_agents.llm.router import LLMRouter

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
    """Return default llama-cpp settings for local runs."""
    return DEFAULT_LLAMA_CPP_SETTINGS


def create_default_llm_client() -> BaseLLMClient:
    """Create an LLM client using package default local llama-cpp settings.

    Returns:
        Base client configured for the default local llama-cpp backend.
    """
    settings = DEFAULT_LLAMA_CPP_SETTINGS
    backend_config = LlamaCppConfig(
        name="llama-local",
        kind="llama_cpp",
        model_path=settings.model,
        hf_model_repo_id=settings.hf_model_repo_id,
        api_model=settings.api_model,
        host=settings.host,
        port=settings.port,
        default_model=settings.api_model,
        model_patterns=(settings.api_model,),
    )
    llama_backend = create_llama_cpp_server(
        model=backend_config.model_path,
        hf_model_repo_id=backend_config.hf_model_repo_id,
        api_model=backend_config.api_model,
        host=backend_config.host,
        port=backend_config.port,
    )
    backend = LlamaCppBackend(
        name=backend_config.name,
        llama_backend=llama_backend,
        default_model=backend_config.api_model,
        config_hash=backend_config_hash(backend_config),
        model_patterns=backend_config.model_patterns,
    )
    router = LLMRouter([backend], default_backend=backend_config.name)
    # Import lazily to avoid cyclic imports with runtime configuration module.
    from design_research_agents.llm.base_client import BaseLLMClient

    return BaseLLMClient(router=router, backend=backend_config.name)
