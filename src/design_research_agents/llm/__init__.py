"""LLM configuration and backend entrypoints for package runtime.

This module keeps legacy backend configuration APIs stable while also exposing
the newer router/config entrypoints for capability-based backend selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .backends.adapters import OpenAIBackendConfig
from .backends.factory import build_backends
from .backends.llama_cpp_server import LlamaCppServerBackend
from .backends.llama_cpp_server import create_backend as create_llama_cpp_server_backend
from .backends.transformers_backend import TransformersBackend
from .backends.transformers_backend import create_backend as create_transformers_backend
from .backends.types import BackendName, parse_backend
from .base_client import BaseLLMClient
from .config import LLMConfig, load_config
from .router import LLMRouter

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
    "LLMRouter",
    "build_backends",
    "configure_llama_cpp_server",
    "configure_openai",
    "configure_router_from_yaml",
    "configure_transformers",
    "load_config",
    "parse_backend",
    "resolve_default_model",
    "shutdown_llama_cpp_server",
    "shutdown_transformers_backend",
]


@dataclass(slots=True)
class _OpenAIConfig:
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    require_api_key: bool = True


_llama_cpp_backend: LlamaCppServerBackend | None = None
_transformers_backend: TransformersBackend | None = None
_openai_config = _OpenAIConfig()
_active_backend: BackendName = "llama-cpp-server"
_default_router: LLMRouter | None = None
_use_router_default: bool = False


def configure_openai(
    *,
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    api_key: str | None = None,
    base_url: str | None = None,
    require_api_key: bool = True,
) -> None:
    """Configure OpenAI defaults and activate the OpenAI backend."""
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

    global _active_backend, _use_router_default
    _openai_config.model = normalized_model
    _openai_config.api_key_env = normalized_api_key_env
    _openai_config.api_key = normalized_api_key
    _openai_config.base_url = normalized_base_url
    _openai_config.require_api_key = require_api_key
    _active_backend = "openai"
    _use_router_default = False


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
    """Configure the llama-cpp backend and activate it for default calls."""
    global _active_backend, _llama_cpp_backend, _use_router_default
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
    _active_backend = "llama-cpp-server"
    _use_router_default = False
    return _llama_cpp_backend


def shutdown_llama_cpp_server() -> None:
    """Stop and clear the configured llama-cpp backend, if present."""
    global _llama_cpp_backend
    if _llama_cpp_backend is None:
        return
    _llama_cpp_backend.close()
    _llama_cpp_backend = None


def configure_transformers(
    *,
    model: str,
    tokenizer: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device: int | str | None = None,
    device_map: str | None = None,
    torch_dtype: object | None = None,
    cache_dir: str | None = None,
    use_fast: bool = True,
    pipeline_task: str = "text-generation",
    model_kwargs: Mapping[str, object] | None = None,
    tokenizer_kwargs: Mapping[str, object] | None = None,
    pipeline_kwargs: Mapping[str, object] | None = None,
    generation_kwargs: Mapping[str, object] | None = None,
) -> TransformersBackend:
    """Configure the Transformers backend and activate it for default calls."""
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model must not be empty.")

    global _active_backend, _transformers_backend, _use_router_default
    shutdown_transformers_backend()
    _transformers_backend = create_transformers_backend(
        model=normalized_model,
        tokenizer=tokenizer,
        revision=revision,
        trust_remote_code=trust_remote_code,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        use_fast=use_fast,
        pipeline_task=pipeline_task,
        model_kwargs=dict(model_kwargs or {}),
        tokenizer_kwargs=dict(tokenizer_kwargs or {}),
        pipeline_kwargs=dict(pipeline_kwargs or {}),
        generation_kwargs=dict(generation_kwargs or {}),
    )
    _active_backend = "transformers"
    _use_router_default = False
    return _transformers_backend


def shutdown_transformers_backend() -> None:
    """Stop and clear the configured Transformers backend, if present."""
    global _transformers_backend
    if _transformers_backend is None:
        return
    _transformers_backend.close()
    _transformers_backend = None


def configure_router_from_yaml(path: str, *, default_backend: str | None = None) -> LLMRouter:
    """Load YAML config, build a router, and register it as runtime default."""
    global _default_router, _use_router_default
    config = load_config(path)
    router = LLMRouter(build_backends(config.backends), default_backend=default_backend)
    _default_router = router
    _use_router_default = True
    return router


def _get_openai_backend_config() -> OpenAIBackendConfig:
    return OpenAIBackendConfig(
        api_key_env=_openai_config.api_key_env,
        api_key=_openai_config.api_key,
        base_url=_openai_config.base_url,
        require_api_key=_openai_config.require_api_key,
    )


def _get_configured_llama_cpp_backend() -> LlamaCppServerBackend | None:
    return _llama_cpp_backend


def _get_configured_transformers_backend() -> TransformersBackend | None:
    return _transformers_backend


def _get_active_backend() -> BackendName:
    return _active_backend


def _get_default_router() -> LLMRouter | None:
    if not _use_router_default:
        return None
    return _default_router


def resolve_default_model(*, backend: str | None = None) -> str:
    """Resolve default model from legacy runtime config or default router."""
    if backend is not None:
        return _resolve_legacy_default_model(parse_backend(backend))

    router = _get_default_router()
    if router is not None:
        return router.default_model()
    return _resolve_legacy_default_model(_active_backend)


def _resolve_legacy_default_model(backend: BackendName) -> str:
    if backend == "openai":
        return _openai_config.model
    if backend == "llama-cpp-server":
        if _llama_cpp_backend is not None:
            return _llama_cpp_backend.api_model
        return "local-model"
    if backend == "transformers":
        if _transformers_backend is not None:
            return _transformers_backend.model
        return "transformers-model"
    if backend == "echo-test":
        return "echo-test-model"
    raise ValueError(f"Unsupported backend '{backend}'.")
