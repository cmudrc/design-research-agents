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
        super().__init__(config=config or ToolRuntimeConfig())


def create_calculator_tool_spec() -> ToolSpec:
    """Return calculator compatibility tool spec from runtime defaults."""
    runtime = BaseToolRuntime()
    for spec in runtime.list_tools():
        if spec.name == "calculator_tool":
            return spec
    raise RuntimeError("calculator_tool is not registered.")


def create_text_stats_tool_spec() -> ToolSpec:
    """Return text-stats compatibility tool spec from runtime defaults."""
    runtime = BaseToolRuntime()
    for spec in runtime.list_tools():
        if spec.name == "text_stats_tool":
            return spec
    raise RuntimeError("text_stats_tool is not registered.")


__all__ = [
    "BaseToolRuntime",
    "create_calculator_tool_spec",
    "create_text_stats_tool_spec",
]
