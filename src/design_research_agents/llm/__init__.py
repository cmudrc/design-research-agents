"""Stable public LLM contracts and client entry points."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Final

from design_research_agents._contracts import LLMMessage, LLMRequest, LLMResponse
from design_research_agents._lazy_exports import module_dir, resolve_lazy_export

_EXPORTS: Final[dict[str, str]] = {
    "AnthropicServiceLLMClient": "design_research_agents.llm.clients:AnthropicServiceLLMClient",
    "AzureOpenAIServiceLLMClient": "design_research_agents.llm.clients:AzureOpenAIServiceLLMClient",
    "DemoLLMClient": "design_research_agents.llm.clients:DemoLLMClient",
    "GeminiServiceLLMClient": "design_research_agents.llm.clients:GeminiServiceLLMClient",
    "GroqServiceLLMClient": "design_research_agents.llm.clients:GroqServiceLLMClient",
    "LlamaCppServerLLMClient": "design_research_agents.llm.clients:LlamaCppServerLLMClient",
    "MLXLocalLLMClient": "design_research_agents.llm.clients:MLXLocalLLMClient",
    "OllamaLLMClient": "design_research_agents.llm.clients:OllamaLLMClient",
    "OpenAICompatibleHTTPLLMClient": "design_research_agents.llm.clients:OpenAICompatibleHTTPLLMClient",
    "OpenAIServiceLLMClient": "design_research_agents.llm.clients:OpenAIServiceLLMClient",
    "SGLangServerLLMClient": "design_research_agents.llm.clients:SGLangServerLLMClient",
    "TransformersLocalLLMClient": "design_research_agents.llm.clients:TransformersLocalLLMClient",
    "VLLMServerLLMClient": "design_research_agents.llm.clients:VLLMServerLLMClient",
}
_SUBMODULES: Final[dict[str, str]] = {
    "clients": "design_research_agents.llm.clients",
    "_backends": "design_research_agents.llm._backends",
}

__all__ = ["LLMMessage", "LLMRequest", "LLMResponse", *_EXPORTS.keys()]


def __getattr__(name: str) -> object:
    """Resolve exported client symbols on first access."""
    submodule_name = _SUBMODULES.get(name)
    if submodule_name is not None:
        value = import_module(submodule_name)
        globals()[name] = value
        return value
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
    from .clients import AnthropicServiceLLMClient as AnthropicServiceLLMClient
    from .clients import AzureOpenAIServiceLLMClient as AzureOpenAIServiceLLMClient
    from .clients import DemoLLMClient as DemoLLMClient
    from .clients import GeminiServiceLLMClient as GeminiServiceLLMClient
    from .clients import GroqServiceLLMClient as GroqServiceLLMClient
    from .clients import LlamaCppServerLLMClient as LlamaCppServerLLMClient
    from .clients import MLXLocalLLMClient as MLXLocalLLMClient
    from .clients import OllamaLLMClient as OllamaLLMClient
    from .clients import OpenAICompatibleHTTPLLMClient as OpenAICompatibleHTTPLLMClient
    from .clients import OpenAIServiceLLMClient as OpenAIServiceLLMClient
    from .clients import SGLangServerLLMClient as SGLangServerLLMClient
    from .clients import TransformersLocalLLMClient as TransformersLocalLLMClient
    from .clients import VLLMServerLLMClient as VLLMServerLLMClient
