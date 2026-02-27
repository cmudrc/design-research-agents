"""Groq service client implementation."""

from __future__ import annotations

from .._backends._providers._groq_service import GroqServiceBackend
from ._shared import _config_hash, _resolve_model_patterns, _SingleBackendLLMClient
from ._snapshot_helpers import groq_service_config_snapshot


class GroqServiceLLMClient(_SingleBackendLLMClient):
    """Client for the official Groq API backend."""

    def __init__(
        self,
        *,
        name: str = "groq",
        default_model: str = "llama-3.1-8b-instant",
        api_key_env: str = "GROQ_API_KEY",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize a Groq service client with sensible defaults."""
        config_hash = _config_hash(
            {
                "kind": "groq_service",
                "name": name,
                "default_model": default_model,
                "api_key_env": api_key_env,
                "api_key": api_key,
                "base_url": base_url,
                "max_retries": max_retries,
            }
        )
        backend = GroqServiceBackend(
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
            config_snapshot=groq_service_config_snapshot(
                api_key_env=api_key_env,
                api_key=api_key,
                base_url=base_url,
            ),
        )


__all__ = ["GroqServiceLLMClient"]
