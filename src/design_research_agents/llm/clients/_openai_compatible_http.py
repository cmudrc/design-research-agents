"""OpenAI-compatible HTTP client implementation."""

from __future__ import annotations

from design_research_agents._contracts._llm import BackendCapabilities

from .._backends._providers._openai_compatible_http import OpenAICompatibleHTTPBackend
from ._shared import _config_hash, _resolve_model_patterns, _SingleBackendLLMClient
from ._snapshot_helpers import openai_compatible_http_config_snapshot

_OPENAI_COMPAT_CAPABILITIES = BackendCapabilities(
    streaming=False,
    tool_calling="best_effort",
    json_mode="prompt+validate",
    vision=False,
    max_context_tokens=None,
)


class OpenAICompatibleHTTPLLMClient(_SingleBackendLLMClient):
    """Client for OpenAI-compatible HTTP endpoints."""

    def __init__(
        self,
        *,
        name: str = "openai-compatible",
        base_url: str = "http://127.0.0.1:8001/v1",
        default_model: str = "qwen2-1.5b-q4",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an OpenAI-compatible HTTP client with sensible defaults."""
        config_hash = _config_hash(
            {
                "kind": "openai_compatible_http",
                "name": name,
                "base_url": base_url,
                "default_model": default_model,
                "api_key_env": api_key_env,
                "api_key": api_key,
                "max_retries": max_retries,
            }
        )
        backend = OpenAICompatibleHTTPBackend(
            name=name,
            base_url=base_url,
            default_model=default_model,
            api_key_env=api_key_env,
            api_key=api_key,
            capabilities=_OPENAI_COMPAT_CAPABILITIES,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(
            backend=backend,
            config_snapshot=openai_compatible_http_config_snapshot(
                api_key_env=api_key_env,
                api_key=api_key,
                base_url=base_url,
            ),
        )


__all__ = ["OpenAICompatibleHTTPLLMClient"]
