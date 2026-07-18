"""Stable public LLM client classes backed by internal provider wrappers."""

from __future__ import annotations

from typing import Final

from design_research_agents._lazy_exports import module_dir, resolve_lazy_export

_EXPORTS: Final[dict[str, str]] = {
    "AnthropicServiceLLMClient": "design_research_agents.llm.clients._anthropic_service:AnthropicServiceLLMClient",
    "AzureOpenAIServiceLLMClient": (
        "design_research_agents.llm.clients._azure_openai_service:AzureOpenAIServiceLLMClient"
    ),
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
