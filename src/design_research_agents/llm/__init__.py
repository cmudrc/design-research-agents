"""LLM routing and client entrypoints."""

from __future__ import annotations

from .backends.factory import build_backends
from .base_client import BaseLLMClient
from .config import LLMConfig, load_config
from .router import LLMRouter

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
    "LLMRouter",
    "build_backends",
    "configure_router_from_yaml",
    "load_config",
    "resolve_default_model",
    "set_default_router",
]

_default_router: LLMRouter | None = None


def configure_router_from_yaml(path: str, *, default_backend: str | None = None) -> LLMRouter:
    """Load YAML config, build a router, and register it as runtime default."""
    config = load_config(path)
    router = LLMRouter(build_backends(config.backends), default_backend=default_backend)
    set_default_router(router)
    return router


def set_default_router(router: LLMRouter | None) -> None:
    """Set or clear the process default router used by ``BaseLLMClient``."""
    global _default_router
    _default_router = router


def _get_default_router() -> LLMRouter | None:
    return _default_router


def resolve_default_model(*, backend: str | None = None) -> str:
    """Resolve a default model from the configured default router."""
    router = _require_default_router()
    if backend is None:
        return router.default_model()

    normalized_backend = backend.strip()
    if not normalized_backend:
        raise ValueError("backend must not be empty when provided.")
    return router.default_model_for_backend(normalized_backend)


def _require_default_router() -> LLMRouter:
    router = _get_default_router()
    if router is None:
        raise ValueError(
            "No default LLM router is configured. "
            "Use configure_router_from_yaml(...) or pass router=... to BaseLLMClient."
        )
    return router
