"""Public package interface for design_research_agents."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .llm import (
    complete,
    configure_llama_cpp_server,
    configure_openai,
    shutdown_llama_cpp_server,
)

# Keep a small, stable public surface for downstream users.
__all__ = [
    "__version__",
    "complete",
    "configure_openai",
    "configure_llama_cpp_server",
    "shutdown_llama_cpp_server",
]

try:
    # Distribution name (from pyproject.toml [project].name)
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    # Running from source without an installed distribution (e.g., direct `python` execution)
    # avoids hard failures in local dev before installation.
    __version__ = "unknown"
