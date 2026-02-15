"""Backend factory for building configured LLM backends."""

from __future__ import annotations

from typing import cast

from design_research_agents.contracts.llm import BackendCapabilities
from design_research_agents.llm.backends.base import BaseLLMBackend
from design_research_agents.llm.backends.providers.echo_test import EchoTestBackend
from design_research_agents.llm.backends.providers.llama_cpp import LlamaCppBackend
from design_research_agents.llm.backends.providers.llama_cpp_server import (
    create_backend as create_llama_cpp_server,
)
from design_research_agents.llm.backends.providers.mlx_local import MlxLocalBackend
from design_research_agents.llm.backends.providers.openai_compatible_http import (
    OpenAICompatibleHTTPBackend,
)
from design_research_agents.llm.backends.providers.openai_service import OpenAIServiceBackend
from design_research_agents.llm.backends.providers.transformers_local import (
    TransformersLocalBackend,
)
from design_research_agents.llm.config import (
    BackendConfig,
    EchoTestConfig,
    LlamaCppConfig,
    MlxLocalConfig,
    OpenAICompatibleHTTPConfig,
    OpenAIServiceConfig,
    TransformersLocalConfig,
    backend_config_hash,
)


def build_backend(config: BackendConfig) -> BaseLLMBackend:
    """Build one concrete backend instance from typed backend config."""
    if isinstance(config, OpenAIServiceConfig):
        capabilities = _resolve_capabilities(config, default=_default_openai_service_caps())
        return OpenAIServiceBackend(
            name=config.name,
            default_model=cast(str, config.default_model),
            api_key_env=config.api_key_env,
            api_key=config.api_key,
            base_url=config.base_url,
            capabilities=capabilities,
            config_hash=backend_config_hash(config),
            max_retries=config.max_retries,
            model_patterns=config.model_patterns,
        )
    if isinstance(config, OpenAICompatibleHTTPConfig):
        capabilities = _resolve_capabilities(config, default=_default_openai_compatible_caps())
        return OpenAICompatibleHTTPBackend(
            name=config.name,
            base_url=config.base_url,
            default_model=cast(str, config.default_model),
            api_key_env=config.api_key_env,
            api_key=config.api_key,
            capabilities=capabilities,
            config_hash=backend_config_hash(config),
            max_retries=config.max_retries,
            model_patterns=config.model_patterns,
        )
    if isinstance(config, TransformersLocalConfig):
        return TransformersLocalBackend(
            name=config.name,
            model_id=config.model_id,
            default_model=cast(str, config.default_model),
            device=config.device,
            dtype=config.dtype,
            quantization=config.quantization,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
            config_hash=backend_config_hash(config),
            max_retries=config.max_retries,
            model_patterns=config.model_patterns,
        )
    if isinstance(config, MlxLocalConfig):
        return MlxLocalBackend(
            name=config.name,
            model_id=config.model_id,
            default_model=cast(str, config.default_model),
            quantization=config.quantization,
            config_hash=backend_config_hash(config),
            max_retries=config.max_retries,
            model_patterns=config.model_patterns,
        )
    if isinstance(config, LlamaCppConfig):
        llama_backend = create_llama_cpp_server(
            model=config.model_path,
            hf_model_repo_id=config.hf_model_repo_id,
            api_model=config.api_model,
            host=config.host,
            port=config.port,
            startup_timeout_seconds=config.startup_timeout_seconds,
            poll_interval_seconds=config.poll_interval_seconds,
            extra_server_args=config.extra_server_args,
        )
        return LlamaCppBackend(
            name=config.name,
            llama_backend=llama_backend,
            default_model=config.api_model,
            config_hash=backend_config_hash(config),
            max_retries=config.max_retries,
            model_patterns=config.model_patterns,
        )
    if isinstance(config, EchoTestConfig):
        return EchoTestBackend(
            name=config.name,
            model=config.model,
            config_hash=backend_config_hash(config),
        )
    raise ValueError(f"Unsupported backend config type: {type(config)}")


def build_backends(configs: tuple[BackendConfig, ...]) -> list[BaseLLMBackend]:
    """Build all backend instances in declaration order."""
    return [build_backend(config) for config in configs]


def _resolve_capabilities(
    config: BackendConfig,
    *,
    default: BackendCapabilities,
) -> BackendCapabilities:
    return config.capabilities.apply(default)


def _default_openai_compatible_caps() -> BackendCapabilities:
    return BackendCapabilities(
        streaming=False,
        tool_calling="best_effort",
        json_mode="prompt+validate",
        vision=False,
        max_context_tokens=None,
    )


def _default_openai_service_caps() -> BackendCapabilities:
    return BackendCapabilities(
        streaming=True,
        tool_calling="native",
        json_mode="native",
        vision=False,
        max_context_tokens=None,
    )
