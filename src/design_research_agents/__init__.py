"""Public package interface for the design-research-agents library.

This module deliberately exposes a compact and stable surface that combines:
- core agent implementations,
- default llama-cpp configuration/client entrypoints, and
- default tool runtime primitives.

Callers can import from this module when they want a single entrypoint that
remains stable even as internal package structure evolves.
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
from .orchestrator import DagOrchestrator, SequentialOrchestrator
from .tools import BaseToolRuntime
from .tracing import TraceConfig, configure_tracing

# Keep a small, stable public surface for downstream users.
__all__ = [
    "AgentRuntime",
    "BaseLLMClient",
    "BaseToolRuntime",
    "DagOrchestrator",
    "DirectLLMAgent",
    "HardwareProfile",
    "LLMRouter",
    "ModelCatalog",
    "ModelSelectionConstraints",
    "ModelSelectionDecision",
    "ModelSelectionIntent",
    "ModelSelectionPolicy",
    "ModelSelectionPolicyConfig",
    "MultiStepAgent",
    "RouterAgent",
    "RuntimeControls",
    "SequentialOrchestrator",
    "SingleStepCodeAgent",
    "ToolCallingAgent",
    "TraceConfig",
    "__version__",
    "configure_router_from_yaml",
    "configure_tracing",
    "create_default_llm_client",
]

try:
    # Distribution name (from pyproject.toml [project].name)
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    # Running from source without an installed distribution (e.g., direct `python` execution)
    # avoids hard failures in local dev before installation.
    __version__ = "unknown"
