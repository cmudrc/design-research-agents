"""Gemini service client implementation."""

from __future__ import annotations

from .._backends._providers._gemini_service import GeminiServiceBackend
from ._shared import _config_hash, _resolve_model_patterns, _SingleBackendLLMClient
from ._snapshot_helpers import gemini_service_config_snapshot


class GeminiServiceLLMClient(_SingleBackendLLMClient):
    """Client for the official Gemini API backend."""

    def __init__(
        self,
        *,
        name: str = "gemini",
        default_model: str = "gemini-2.5-flash",
        api_key_env: str = "GOOGLE_API_KEY",
        api_key: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize a Gemini service client with sensible defaults."""
        config_hash = _config_hash(
            {
                "kind": "gemini_service",
                "name": name,
                "default_model": default_model,
                "api_key_env": api_key_env,
                "api_key": api_key,
                "max_retries": max_retries,
            }
        )
        backend = GeminiServiceBackend(
            name=name,
            default_model=default_model,
            api_key_env=api_key_env,
            api_key=api_key,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(
            backend=backend,
            config_snapshot=gemini_service_config_snapshot(api_key_env=api_key_env, api_key=api_key),
        )


__all__ = ["GeminiServiceLLMClient"]
