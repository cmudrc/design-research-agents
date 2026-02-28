"""Stable public LLM contracts and client entry points."""

from design_research_agents._contracts import LLMMessage, LLMRequest, LLMResponse

from .clients import (
    AnthropicServiceLLMClient,
    AzureOpenAIServiceLLMClient,
    GeminiServiceLLMClient,
    GroqServiceLLMClient,
    LlamaCppServerLLMClient,
    MLXLocalLLMClient,
    OllamaLLMClient,
    OpenAICompatibleHTTPLLMClient,
    OpenAIServiceLLMClient,
    SGLangServerLLMClient,
    TransformersLocalLLMClient,
    VLLMServerLLMClient,
)

__all__ = [
    "AnthropicServiceLLMClient",
    "AzureOpenAIServiceLLMClient",
    "GeminiServiceLLMClient",
    "GroqServiceLLMClient",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LlamaCppServerLLMClient",
    "MLXLocalLLMClient",
    "OllamaLLMClient",
    "OpenAICompatibleHTTPLLMClient",
    "OpenAIServiceLLMClient",
    "SGLangServerLLMClient",
    "TransformersLocalLLMClient",
    "VLLMServerLLMClient",
]
