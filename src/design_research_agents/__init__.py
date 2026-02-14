"""Public package interface for design_research_agents."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .agent import MultiStepAgent, RouterAgent, SingleStepCodeAgent, ToolCallingAgent
from .contracts import (
    Agent,
    AgentResult,
    AgentStreamEvent,
    LLMAuthError,
    LLMChatParams,
    LLMClient,
    LLMError,
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderAdapter,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMRole,
    LLMStreamEvent,
    LLMStreamEventKind,
    ToolCostHints,
    ToolResult,
    ToolRuntime,
    ToolSpec,
)
from .llm import (
    BaseLLMClient,
    complete,
    configure_llama_cpp_server,
    configure_openai,
    shutdown_llama_cpp_server,
)
from .schemas import SCHEMA_NAMES, SCHEMA_VERSION, load_schema
from .tools import BaseToolRuntime, create_calculator_tool_spec, create_text_stats_tool_spec

# Keep a small, stable public surface for downstream users.
__all__ = [
    "__version__",
    "Agent",
    "AgentResult",
    "AgentStreamEvent",
    "MultiStepAgent",
    "RouterAgent",
    "SingleStepCodeAgent",
    "ToolCallingAgent",
    "BaseLLMClient",
    "BaseToolRuntime",
    "LLMAuthError",
    "LLMChatParams",
    "LLMClient",
    "LLMError",
    "LLMInvalidRequestError",
    "LLMMessage",
    "LLMProviderAdapter",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMRole",
    "LLMStreamEvent",
    "LLMStreamEventKind",
    "SCHEMA_NAMES",
    "SCHEMA_VERSION",
    "ToolCostHints",
    "ToolResult",
    "ToolRuntime",
    "ToolSpec",
    "complete",
    "create_calculator_tool_spec",
    "create_text_stats_tool_spec",
    "configure_openai",
    "configure_llama_cpp_server",
    "load_schema",
    "shutdown_llama_cpp_server",
]

try:
    # Distribution name (from pyproject.toml [project].name)
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    # Running from source without an installed distribution (e.g., direct `python` execution)
    # avoids hard failures in local dev before installation.
    __version__ = "unknown"
