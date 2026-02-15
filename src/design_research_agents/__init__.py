"""Public package interface for the design-research-agents library.

This module deliberately exposes a compact and stable surface that combines:
- core agent implementations,
- workflow runtime primitives,
- default llama-cpp configuration/client entrypoints, and
- default tool runtime primitives.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .agent import (
    AgentRuntime,
    DirectLLMAgent,
    MultiStepAgent,
    RouterAgent,
    RuntimeControls,
    SingleStepCodeAgent,
    ToolCallingAgent,
)
from .contracts.orchestrator import AgentStep, LogicStep, ToolStep
from .llm import BaseLLMClient, LLMRouter, configure_router_from_yaml
from .llm.backends.default import create_default_llm_client
from .model_selection import (
    HardwareProfile,
    ModelCatalog,
    ModelSelectionConstraints,
    ModelSelectionDecision,
    ModelSelectionIntent,
    ModelSelectionPolicy,
    ModelSelectionPolicyConfig,
)
from .orchestrator import WorkflowRuntime
from .tools import BaseToolRuntime, ToolRuntimeConfig, UnifiedToolRuntime, load_tool_runtime_config
from .tracing import TraceConfig, configure_tracing

__all__ = [
    "AgentRuntime",
    "AgentStep",
    "BaseLLMClient",
    "BaseToolRuntime",
    "DirectLLMAgent",
    "HardwareProfile",
    "LLMRouter",
    "LogicStep",
    "ModelCatalog",
    "ModelSelectionConstraints",
    "ModelSelectionDecision",
    "ModelSelectionIntent",
    "ModelSelectionPolicy",
    "ModelSelectionPolicyConfig",
    "MultiStepAgent",
    "RouterAgent",
    "RuntimeControls",
    "SingleStepCodeAgent",
    "ToolCallingAgent",
    "ToolRuntimeConfig",
    "ToolStep",
    "TraceConfig",
    "UnifiedToolRuntime",
    "WorkflowRuntime",
    "__version__",
    "configure_router_from_yaml",
    "configure_tracing",
    "create_default_llm_client",
    "load_tool_runtime_config",
]

try:
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    __version__ = "unknown"
