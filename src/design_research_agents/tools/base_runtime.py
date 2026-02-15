"""Compatibility shim over the unified source-based tool runtime.

Historically this module exposed an in-memory runtime with two built-in tools.
The new implementation keeps the same public class name while delegating to
``UnifiedToolRuntime`` and preserving ``register_tool`` customization.
"""

from __future__ import annotations

from design_research_agents.contracts.tools import ToolSpec

from .config import ToolRuntimeConfig
from .runtime import UnifiedToolRuntime


class BaseToolRuntime(UnifiedToolRuntime):
    """Backward-compatible runtime alias backed by ``UnifiedToolRuntime``."""

    def __init__(self, *, config: ToolRuntimeConfig | None = None) -> None:
        """Initialize compatibility runtime with optional config override."""
        super().__init__(config=config or ToolRuntimeConfig())


def create_calculator_spec() -> ToolSpec:
    """Return calculator tool spec from runtime defaults."""
    runtime = BaseToolRuntime()
    for spec in runtime.list_tools():
        if spec.name == "calculator":
            return spec
    raise RuntimeError("calculator is not registered.")


def create_text_word_count_spec() -> ToolSpec:
    """Return text word-count tool spec from runtime defaults."""
    runtime = BaseToolRuntime()
    for spec in runtime.list_tools():
        if spec.name == "text.word_count":
            return spec
    raise RuntimeError("text.word_count is not registered.")


__all__ = [
    "BaseToolRuntime",
    "create_calculator_spec",
    "create_text_word_count_spec",
]
