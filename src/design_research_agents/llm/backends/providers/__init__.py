"""Provider implementations for concrete LLM backend integrations."""

from .echo_test import EchoTestBackend
from .llama_cpp import LlamaCppBackend
from .llama_cpp_server import LlamaCppServerBackend, create_backend
from .mlx_local import MlxLocalBackend
from .ollama_local import OllamaLocalBackend
from .ollama_server import OllamaServerBackend
from .ollama_server import create_backend as create_ollama_backend
from .openai_compatible_http import OpenAICompatibleHTTPBackend
from .openai_service import OpenAIServiceBackend
from .sglang_local import SglangLocalBackend
from .sglang_server import SglangServerBackend
from .sglang_server import create_backend as create_sglang_backend
from .transformers_local import TransformersLocalBackend
from .vllm_local import VllmLocalBackend
from .vllm_server import VllmServerBackend
from .vllm_server import create_backend as create_vllm_backend

__all__ = [
    "EchoTestBackend",
    "LlamaCppBackend",
    "LlamaCppServerBackend",
    "MlxLocalBackend",
    "OllamaLocalBackend",
    "OllamaServerBackend",
    "OpenAICompatibleHTTPBackend",
    "OpenAIServiceBackend",
    "SglangLocalBackend",
    "SglangServerBackend",
    "TransformersLocalBackend",
    "VllmLocalBackend",
    "VllmServerBackend",
    "create_backend",
    "create_ollama_backend",
    "create_sglang_backend",
    "create_vllm_backend",
]
