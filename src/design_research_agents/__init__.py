"""Public package interface for the design-research-agents library.

This module deliberately exposes a compact and stable surface that combines:
- agent implementations,
- protocol/contract types,
- backend configuration helpers,
- prompt/schema loaders, and
- default tool runtime primitives.

Callers can import from this module when they want a single entrypoint that
remains stable even as internal package structure evolves.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .agent import (
    DirectLLMAgent,
    MultiStepAgent,
    RouterAgent,
    SingleStepCodeAgent,
    ToolCallingAgent,
)
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
    resolve_default_model,
    shutdown_llama_cpp_server,
)
from .prompts import PROMPT_NAMES, load_prompt, render_prompt
from .schemas import SCHEMA_NAMES, SCHEMA_VERSION, load_schema
from .tools import (
    BaseToolRuntime,
    create_calculator_tool_spec,
    create_text_stats_tool_spec,
)

# Keep a small, stable public surface for downstream users.
__all__ = [
    "PROMPT_NAMES",
    "SCHEMA_NAMES",
    "SCHEMA_VERSION",
    "Agent",
    "AgentResult",
    "AgentStreamEvent",
    "BaseLLMClient",
    "BaseToolRuntime",
    "DirectLLMAgent",
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
    "MultiStepAgent",
    "RouterAgent",
    "SingleStepCodeAgent",
    "ToolCallingAgent",
    "ToolCostHints",
    "ToolResult",
    "ToolRuntime",
    "ToolSpec",
    "__version__",
    "complete",
    "configure_llama_cpp_server",
    "configure_openai",
    "create_calculator_tool_spec",
    "create_text_stats_tool_spec",
    "load_prompt",
    "load_schema",
    "render_prompt",
    "resolve_default_model",
    "shutdown_llama_cpp_server",
]

try:
    # Distribution name (from pyproject.toml [project].name)
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    # Running from source without an installed distribution (e.g., direct `python` execution)
    # avoids hard failures in local dev before installation.
    __version__ = "unknown"
