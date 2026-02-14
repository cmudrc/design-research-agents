"""Backend adapter exports and backend-construction helpers.

This namespace re-exports adapter classes and backend factory functions used by
the higher-level LLM client/runtime configuration modules.
"""

from .adapters import (
    EchoTestProviderAdapter,
    LlamaCppServerProviderAdapter,
    OpenAIBackendConfig,
    OpenAIProviderAdapter,
    build_backend_adapter,
)
from .default import (
    DEFAULT_LLAMA_CPP_SETTINGS,
    DefaultLlamaCppSettings,
    configure_default_llama_cpp_backend,
    create_default_llm_client,
)
from .llama_cpp_server import (
    LlamaCppServerBackend,
)
from .llama_cpp_server import (
    create_backend as create_llama_cpp_server_backend,
)

__all__ = [
    "DEFAULT_LLAMA_CPP_SETTINGS",
    "DefaultLlamaCppSettings",
    "EchoTestProviderAdapter",
    "LlamaCppServerBackend",
    "LlamaCppServerProviderAdapter",
    "OpenAIBackendConfig",
    "OpenAIProviderAdapter",
    "build_backend_adapter",
    "configure_default_llama_cpp_backend",
    "create_default_llm_client",
    "create_llama_cpp_server_backend",
]
