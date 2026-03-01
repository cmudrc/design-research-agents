"""OpenAI service client implementation."""

from __future__ import annotations

from .._backends._providers._openai_service import OpenAIServiceBackend
from ._shared import _config_hash, _resolve_model_patterns, _SingleBackendLLMClient
from ._snapshot_helpers import openai_service_config_snapshot


class OpenAIServiceLLMClient(_SingleBackendLLMClient):
    """Client for the official OpenAI API backend."""

    def __init__(
        self,
        *,
        name: str = "openai",
        default_model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an OpenAI service client with sensible defaults."""
        config_hash = _config_hash(
            {
                "kind": "openai_service",
                "name": name,
                "default_model": default_model,
                "api_key_env": api_key_env,
                "api_key": api_key,
                "base_url": base_url,
                "max_retries": max_retries,
            }
        )
        backend = OpenAIServiceBackend(
            name=name,
            default_model=default_model,
            api_key_env=api_key_env,
            api_key=api_key,
            base_url=base_url,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(
            backend=backend,
            config_snapshot=openai_service_config_snapshot(api_key_env=api_key_env, api_key=api_key),
        )


__all__ = ["OpenAIServiceLLMClient"]
