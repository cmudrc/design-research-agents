"""Minimal interface for demo LLM backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .backends.llama_cpp_server import (
    LlamaCppServerBackend,
)
from .backends.llama_cpp_server import (
    create_backend as create_llama_cpp_server_backend,
)
from .backends.local_openai import complete as local_complete
from .backends.openai import complete as openai_complete
from .backends.types import BackendName, parse_backend

__all__ = [
    "complete",
    "configure_openai",
    "configure_llama_cpp_server",
    "shutdown_llama_cpp_server",
]


@dataclass(slots=True)
class _OpenAIConfig:
    """In-process defaults used for the OpenAI backend."""

    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    require_api_key: bool = True


# Process-wide singleton so llama server persists across repeated calls.
_llama_cpp_backend: LlamaCppServerBackend | None = None
_openai_config = _OpenAIConfig()


def configure_openai(
    *,
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    api_key: str | None = None,
    base_url: str | None = None,
    require_api_key: bool = True,
) -> None:
    """Configure in-process defaults for ``backend='openai'`` calls.

    Args:
        model: OpenAI model name used by default.
        api_key_env: Environment variable name used for API key lookup.
        api_key: Explicit API key value. When provided, it overrides ``api_key_env``.
        base_url: Optional OpenAI-compatible API base URL.
        require_api_key: Whether missing API keys should raise an error.

    Raises:
        ValueError: If required string parameters are empty.
    """
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

    _openai_config.model = normalized_model
    _openai_config.api_key_env = normalized_api_key_env
    _openai_config.api_key = normalized_api_key
    _openai_config.base_url = normalized_base_url
    _openai_config.require_api_key = require_api_key


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
) -> LlamaCppServerBackend:
    """Configure the process-managed llama-cpp backend used by ``complete``.

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
    global _llama_cpp_backend
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
    return _llama_cpp_backend


def shutdown_llama_cpp_server() -> None:
    """Stop and clear the configured llama-cpp backend, if present."""
    global _llama_cpp_backend
    if _llama_cpp_backend is None:
        # Idempotent shutdown keeps callers from needing extra state checks.
        return
    _llama_cpp_backend.close()
    _llama_cpp_backend = None


def complete(prompt: str, backend: BackendName = "local") -> str:
    """Generate text with the selected backend.

    Args:
        prompt: Prompt text sent to the selected backend.
        backend: Backend name.

    Returns:
        Generated response text from the selected backend.

    Raises:
        RuntimeError: If the selected backend is missing required setup.
        ValueError: If ``backend`` is not supported.
    """
    # Normalizes user/CLI input and validates supported names.
    selected_backend = parse_backend(backend)
    if selected_backend == "local":
        return local_complete(prompt)
    if selected_backend == "openai":
        # Standard hosted provider call path.
        return openai_complete(
            prompt,
            model=_openai_config.model,
            api_key_env=_openai_config.api_key_env,
            api_key=_openai_config.api_key,
            base_url=_openai_config.base_url,
            require_api_key=_openai_config.require_api_key,
        )
    if selected_backend == "llama-cpp-server":
        if _llama_cpp_backend is None:
            raise RuntimeError(
                "llama-cpp-server backend is not configured. "
                "Call configure_llama_cpp_server(model=...) before use."
            )
        # Delegate completion to the process-managed wrapper.
        return _llama_cpp_backend.complete(prompt)

    # Keep an explicit guard for mypy/pyright exhaustiveness.
    raise ValueError(f"Unsupported backend '{backend}'.")
