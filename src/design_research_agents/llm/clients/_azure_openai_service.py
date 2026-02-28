"""Azure OpenAI service client implementation."""

from __future__ import annotations

from .._backends._providers._azure_openai_service import AzureOpenAIServiceBackend
from ._shared import _config_hash, _resolve_model_patterns, _SingleBackendLLMClient
from ._snapshot_helpers import azure_openai_service_config_snapshot


class AzureOpenAIServiceLLMClient(_SingleBackendLLMClient):
    """Client for the Azure OpenAI API via the official OpenAI SDK."""

    def __init__(
        self,
        *,
        name: str = "azure-openai",
        default_model: str = "gpt-4o-mini",
        api_key_env: str = "AZURE_OPENAI_API_KEY",
        api_key: str | None = None,
        azure_endpoint_env: str = "AZURE_OPENAI_ENDPOINT",
        azure_endpoint: str | None = None,
        api_version_env: str = "AZURE_OPENAI_API_VERSION",
        api_version: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an Azure OpenAI service client with sensible defaults."""
        config_hash = _config_hash(
            {
                "kind": "azure_openai_service",
                "name": name,
                "default_model": default_model,
                "api_key_env": api_key_env,
                "api_key": api_key,
                "azure_endpoint_env": azure_endpoint_env,
                "azure_endpoint": azure_endpoint,
                "api_version_env": api_version_env,
                "api_version": api_version,
                "max_retries": max_retries,
            }
        )
        backend = AzureOpenAIServiceBackend(
            name=name,
            default_model=default_model,
            api_key_env=api_key_env,
            api_key=api_key,
            azure_endpoint_env=azure_endpoint_env,
            azure_endpoint=azure_endpoint,
            api_version_env=api_version_env,
            api_version=api_version,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(
            backend=backend,
            config_snapshot=azure_openai_service_config_snapshot(
                api_key_env=api_key_env,
                api_key=api_key,
                azure_endpoint_env=azure_endpoint_env,
                azure_endpoint=azure_endpoint,
                api_version_env=api_version_env,
                api_version=api_version,
            ),
        )


__all__ = ["AzureOpenAIServiceLLMClient"]
