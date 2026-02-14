"""Tool runtime implementations and default tool specification helpers.

The exported runtime is intentionally lightweight and in-memory so examples and
tests can run without external infrastructure. The helper constructors expose
default tool specifications used by agents and integration tests.
"""

from .base_runtime import BaseToolRuntime, create_calculator_tool_spec, create_text_stats_tool_spec

__all__ = ["BaseToolRuntime", "create_calculator_tool_spec", "create_text_stats_tool_spec"]
