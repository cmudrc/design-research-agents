"""Public package interface grouped by strict facade namespaces."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._public_api import agents, contracts, llm, mcp, models, schemas, tools, tracing, workflows

__all__ = [
    "__version__",
    "agents",
    "contracts",
    "llm",
    "mcp",
    "models",
    "schemas",
    "tools",
    "tracing",
    "workflows",
]

try:
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    __version__ = "unknown"
