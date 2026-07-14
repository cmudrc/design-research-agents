"""Static public interface for LLM contracts and lazy client exports."""

from design_research_agents._contracts import LLMMessage as LLMMessage
from design_research_agents._contracts import LLMRequest as LLMRequest
from design_research_agents._contracts import LLMResponse as LLMResponse
from design_research_agents.llm.clients import AnthropicServiceLLMClient as AnthropicServiceLLMClient
from design_research_agents.llm.clients import (
    AzureOpenAIServiceLLMClient as AzureOpenAIServiceLLMClient,
)
from design_research_agents.llm.clients import GeminiServiceLLMClient as GeminiServiceLLMClient
from design_research_agents.llm.clients import GroqServiceLLMClient as GroqServiceLLMClient
from design_research_agents.llm.clients import LlamaCppServerLLMClient as LlamaCppServerLLMClient
from design_research_agents.llm.clients import MLXLocalLLMClient as MLXLocalLLMClient
from design_research_agents.llm.clients import OllamaLLMClient as OllamaLLMClient
from design_research_agents.llm.clients import (
    OpenAICompatibleHTTPLLMClient as OpenAICompatibleHTTPLLMClient,
)
from design_research_agents.llm.clients import OpenAIServiceLLMClient as OpenAIServiceLLMClient
from design_research_agents.llm.clients import SGLangServerLLMClient as SGLangServerLLMClient
from design_research_agents.llm.clients import (
    TransformersLocalLLMClient as TransformersLocalLLMClient,
)
from design_research_agents.llm.clients import VLLMServerLLMClient as VLLMServerLLMClient

from . import clients as clients

__all__: list[str]
