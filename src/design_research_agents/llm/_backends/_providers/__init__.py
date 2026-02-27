"""Provider implementations for concrete LLM backend integrations."""

from ._anthropic_service import AnthropicServiceBackend
from ._echo_test import EchoTestBackend
from ._gemini_service import GeminiServiceBackend
from ._groq_service import GroqServiceBackend
from ._llama_cpp import LlamaCppBackend
from ._llama_cpp_server import LlamaCppServerBackend, create_backend
from ._mlx_local import MlxLocalBackend
from ._ollama_local import OllamaLocalBackend
from ._ollama_server import OllamaServerBackend
from ._ollama_server import create_backend as create_ollama_backend
from ._openai_compatible_http import OpenAICompatibleHTTPBackend
from ._openai_service import OpenAIServiceBackend
from ._sglang_local import SglangLocalBackend
from ._sglang_server import SglangServerBackend
from ._sglang_server import create_backend as create_sglang_backend
from ._transformers_local import TransformersLocalBackend
from ._vllm_local import VllmLocalBackend
from ._vllm_server import VllmServerBackend
from ._vllm_server import create_backend as create_vllm_backend

__all__ = [
    "AnthropicServiceBackend",
    "EchoTestBackend",
    "GeminiServiceBackend",
    "GroqServiceBackend",
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
