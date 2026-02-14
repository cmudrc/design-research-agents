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
    DirectLLMAgent,
    MultiStepAgent,
    RouterAgent,
    SingleStepCodeAgent,
    ToolCallingAgent,
)
from .llm import configure_llama_cpp_server
from .llm.backends.default import create_default_llm_client
from .tools import BaseToolRuntime

# Backward/ergonomic alias requested for top-level llama-cpp default client creation.
create_defult_llama_cpp_client = create_default_llm_client

# Keep a small, stable public surface for downstream users.
__all__ = [
    "BaseToolRuntime",
    "DirectLLMAgent",
    "MultiStepAgent",
    "RouterAgent",
    "SingleStepCodeAgent",
    "ToolCallingAgent",
    "__version__",
    "configure_llama_cpp_server",
    "create_default_llm_client",
]

try:
    # Distribution name (from pyproject.toml [project].name)
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    # Running from source without an installed distribution (e.g., direct `python` execution)
    # avoids hard failures in local dev before installation.
    __version__ = "unknown"
