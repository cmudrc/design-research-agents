"""LLM configuration and backend entrypoints for the package runtime.

This module owns process-wide backend configuration, default model resolution,
and backend routing helpers. It acts as the bridge between high-level agent
code and backend-specific implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .backends.adapters import OpenAIBackendConfig
from .backends.llama_cpp_server import (
    LlamaCppServerBackend,
)
from .backends.llama_cpp_server import (
    create_backend as create_llama_cpp_server_backend,
)
from .backends.types import BackendName, parse_backend
from .base_client import BaseLLMClient

__all__ = [
    "BaseLLMClient",
    "configure_llama_cpp_server",
    "configure_openai",
    "parse_backend",
    "resolve_default_model",
    "shutdown_llama_cpp_server",
]


@dataclass(slots=True)
class _OpenAIConfig:
    """In-process defaults used for the OpenAI backend integration.

    Instances of this dataclass represent mutable process-level configuration
    that is copied into immutable adapter config objects per request.
    """

    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    require_api_key: bool = True


# Process-wide singleton so llama server persists across repeated calls.
_llama_cpp_backend: LlamaCppServerBackend | None = None
_openai_config = _OpenAIConfig()
# Active backend powers "set it once, use it everywhere" call paths.
_active_backend: BackendName = "llama-cpp-server"


def configure_openai(
    *,
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    api_key: str | None = None,
    base_url: str | None = None,
    require_api_key: bool = True,
) -> None:
    """Configure OpenAI defaults and activate the OpenAI backend.

    Args:
        model: OpenAI model name used by default.
        api_key_env: Environment variable name used for API key lookup.
        api_key: Explicit API key value. When provided, it overrides ``api_key_env``.
        base_url: Optional OpenAI-compatible API base URL.
        require_api_key: Whether missing API keys should raise an error.

    Raises:
        ValueError: If required string parameters are empty.
    """
    # Normalize once so all call paths inherit clean defaults.
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model must not be empty.")

    normalized_api_key_env = api_key_env.strip()
    if not normalized_api_key_env:
        raise ValueError("api_key_env must not be empty.")

    normalized_api_key = api_key.strip() if api_key is not None else None
    if normalized_api_key == "":
        normalized_api_key = None

    normalized_base_url = base_url.strip() if base_url is not None else None
    if normalized_base_url == "":
        normalized_base_url = None

    global _active_backend

    _openai_config.model = normalized_model
    _openai_config.api_key_env = normalized_api_key_env
    _openai_config.api_key = normalized_api_key
    _openai_config.base_url = normalized_base_url
    _openai_config.require_api_key = require_api_key
    # Configure selects OpenAI as the default backend for later calls.
    _active_backend = "openai"


def configure_llama_cpp_server(
    model: str,
    *,
    hf_model_repo_id: str | None = None,
    api_model: str = "local-model",
    host: str = "127.0.0.1",
    port: int = 8001,
    startup_timeout_seconds: float = 60.0,
    poll_interval_seconds: float = 0.25,
    extra_server_args: Sequence[str] = (),
) -> LlamaCppServerBackend | None:
    """Configure the llama-cpp backend and activate it for default calls.

    Args:
        model: ``llama_cpp.server`` ``--model`` value.
        hf_model_repo_id: Optional Hugging Face repository id.
        api_model: OpenAI-compatible model identifier used for completions.
        host: Host used by the local server.
        port: Port used by the local server.
        startup_timeout_seconds: Max startup wait duration.
        poll_interval_seconds: Delay between readiness checks.
        extra_server_args: Extra CLI arguments for ``llama_cpp.server``.

    Returns:
        Configured backend instance that will be reused across calls.
    """
    global _active_backend, _llama_cpp_backend
    # Reconfiguration replaces any existing managed server instance.
    shutdown_llama_cpp_server()
    _llama_cpp_backend = create_llama_cpp_server_backend(
        model=model,
        hf_model_repo_id=hf_model_repo_id,
        api_model=api_model,
        host=host,
        port=port,
        startup_timeout_seconds=startup_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        extra_server_args=extra_server_args,
    )
    # Configure selects llama-cpp-server as the default backend for later calls.
    _active_backend = "llama-cpp-server"
    return _llama_cpp_backend


def shutdown_llama_cpp_server() -> None:
    """Stop and clear the configured llama-cpp backend, if present.

    The function is idempotent and safe to call even when no backend has been
    configured yet.
    """
    global _llama_cpp_backend
    if _llama_cpp_backend is None:
        # Idempotent shutdown keeps callers from needing extra state checks.
        return
    _llama_cpp_backend.close()
    _llama_cpp_backend = None


def _get_openai_backend_config() -> OpenAIBackendConfig:
    """Return a snapshot of the current process-wide OpenAI configuration.

    Returning a value object prevents downstream mutation of global settings.

    Returns:
        Immutable OpenAI backend configuration snapshot.
    """
    # Return a value object so adapters cannot mutate process-wide config by accident.
    return OpenAIBackendConfig(
        api_key_env=_openai_config.api_key_env,
        api_key=_openai_config.api_key,
        base_url=_openai_config.base_url,
        require_api_key=_openai_config.require_api_key,
    )


def _get_configured_llama_cpp_backend() -> LlamaCppServerBackend | None:
    """Return currently configured llama-cpp backend instance, if any.

    Internal helper used by adapter resolution paths.

    Returns:
        Configured llama-cpp backend instance, if available.
    """
    return _llama_cpp_backend


def _get_active_backend() -> BackendName:
    """Return process-wide active backend used for default completions.

    Internal helper consumed by ``BaseLLMClient`` when no backend override is set.

    Returns:
        Backend name selected for default calls.
    """
    return _active_backend


def resolve_default_model(*, backend: BackendName | None = None) -> str:
    """Resolve the default model name from configured backend state.

    Args:
        backend: Optional backend override. Uses active backend when omitted.

    Returns:
        Default model name for the selected backend.
    """
    selected_backend = _active_backend if backend is None else parse_backend(backend)
    if selected_backend == "openai":
        return _openai_config.model
    if selected_backend == "llama-cpp-server":
        if _llama_cpp_backend is not None:
            return _llama_cpp_backend.api_model
        # Keep a stable fallback when llama backend has not been configured yet.
        return "local-model"
    if selected_backend == "echo-test":
        return "echo-test-model"
    raise ValueError(f"Unsupported backend '{selected_backend}'.")
