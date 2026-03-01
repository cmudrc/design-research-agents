"""Azure OpenAI service backend built on the official OpenAI SDK."""

from __future__ import annotations

import os
from typing import Any, override

from design_research_agents._contracts._llm import BackendCapabilities, BackendStatus

from ._openai_service import OpenAIServiceBackend


class AzureOpenAIServiceBackend(OpenAIServiceBackend):
    """Backend that calls Azure OpenAI via the official OpenAI SDK."""

    @override
    def __init__(
        self,
        *,
        name: str,
        default_model: str,
        api_key_env: str,
        api_key: str | None,
        azure_endpoint_env: str,
        azure_endpoint: str | None,
        api_version_env: str,
        api_version: str | None,
        capabilities: BackendCapabilities | None = None,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
    ) -> None:
        """Configure Azure OpenAI client defaults and optional capability overrides."""
        super().__init__(
            name=name,
            default_model=default_model,
            api_key_env=api_key_env,
            api_key=api_key,
            base_url=azure_endpoint,
            capabilities=capabilities,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self.kind = "azure_openai_service"
        self._azure_endpoint_env = azure_endpoint_env
        self._azure_endpoint = azure_endpoint
        self._api_version_env = api_version_env
        self._api_version = api_version

    @override
    def healthcheck(self) -> BackendStatus:
        """Return static status for a configured Azure OpenAI backend."""
        return BackendStatus(ok=True, message="Azure OpenAI backend configured.")

    @override
    def _create_client(self) -> Any:
        """Create and cache the Azure OpenAI SDK client instance."""
        api_key = self._resolve_api_key()
        azure_endpoint = self._resolve_azure_endpoint()
        api_version = self._resolve_api_version()
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for azure_openai_service backends. Install with: pip install -e ."
            ) from exc
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return self._client

    def _resolve_azure_endpoint(self) -> str:
        """Resolve Azure endpoint from explicit value or configured environment variable."""
        if self._azure_endpoint:
            return self._azure_endpoint
        env_value = os.getenv(self._azure_endpoint_env)
        if env_value:
            return env_value
        raise RuntimeError(f"{self._azure_endpoint_env} is not set.")

    def _resolve_api_version(self) -> str:
        """Resolve Azure API version from explicit value or configured environment variable."""
        if self._api_version:
            return self._api_version
        env_value = os.getenv(self._api_version_env)
        if env_value:
            return env_value
        raise RuntimeError(f"{self._api_version_env} is not set.")


__all__ = ["AzureOpenAIServiceBackend"]
