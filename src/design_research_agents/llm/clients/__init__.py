"""Stable public LLM client classes backed by internal provider wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from design_research_agents._lazy_exports import module_dir, resolve_lazy_export

_EXPORTS: Final[dict[str, str]] = {
    "AnthropicServiceLLMClient": "design_research_agents.llm.clients._anthropic_service:AnthropicServiceLLMClient",
    "AzureOpenAIServiceLLMClient": (
        "design_research_agents.llm.clients._azure_openai_service:AzureOpenAIServiceLLMClient"
    ),
    "DemoLLMClient": "design_research_agents.llm.clients._demo:DemoLLMClient",
    "GeminiServiceLLMClient": "design_research_agents.llm.clients._gemini_service:GeminiServiceLLMClient",
    "GroqServiceLLMClient": "design_research_agents.llm.clients._groq_service:GroqServiceLLMClient",
    "LlamaCppServerLLMClient": "design_research_agents.llm.clients._llama_cpp_server:LlamaCppServerLLMClient",
    "MLXLocalLLMClient": "design_research_agents.llm.clients._mlx_local:MLXLocalLLMClient",
    "OllamaLLMClient": "design_research_agents.llm.clients._ollama:OllamaLLMClient",
    "OpenAICompatibleHTTPLLMClient": (
        "design_research_agents.llm.clients._openai_compatible_http:OpenAICompatibleHTTPLLMClient"
    ),
    "OpenAIServiceLLMClient": "design_research_agents.llm.clients._openai_service:OpenAIServiceLLMClient",
    "SGLangServerLLMClient": "design_research_agents.llm.clients._sglang_server:SGLangServerLLMClient",
    "TransformersLocalLLMClient": "design_research_agents.llm.clients._transformers_local:TransformersLocalLLMClient",
    "VLLMServerLLMClient": "design_research_agents.llm.clients._vllm_server:VLLMServerLLMClient",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> object:
    """Resolve exported client symbols on first access."""
    return resolve_lazy_export(
        module_name=__name__,
        exports=_EXPORTS,
        export_name=name,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    """Return module attributes, including lazy exports."""
    return module_dir(globals(), __all__)


if TYPE_CHECKING:
    from ._anthropic_service import AnthropicServiceLLMClient as AnthropicServiceLLMClient
    from ._azure_openai_service import AzureOpenAIServiceLLMClient as AzureOpenAIServiceLLMClient
    from ._demo import DemoLLMClient as DemoLLMClient
    from ._gemini_service import GeminiServiceLLMClient as GeminiServiceLLMClient
    from ._groq_service import GroqServiceLLMClient as GroqServiceLLMClient
    from ._llama_cpp_server import LlamaCppServerLLMClient as LlamaCppServerLLMClient
    from ._mlx_local import MLXLocalLLMClient as MLXLocalLLMClient
    from ._ollama import OllamaLLMClient as OllamaLLMClient
    from ._openai_compatible_http import OpenAICompatibleHTTPLLMClient as OpenAICompatibleHTTPLLMClient
    from ._openai_service import OpenAIServiceLLMClient as OpenAIServiceLLMClient
    from ._sglang_server import SGLangServerLLMClient as SGLangServerLLMClient
    from ._transformers_local import TransformersLocalLLMClient as TransformersLocalLLMClient
    from ._vllm_server import VLLMServerLLMClient as VLLMServerLLMClient
