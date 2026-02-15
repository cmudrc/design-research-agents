"""Manifest-less lazy tool parsing, discovery, and execution."""

from .discovery import LazyDiscoveryDiagnostic, LazyToolDefinition, discover_lazy_tools
from .parser import LazyHeaderError, LazyToolHeader, parse_lazy_tool_header
from .runner import LazyToolRuntimeError, run_lazy_tool

__all__ = [
    "LazyDiscoveryDiagnostic",
    "LazyHeaderError",
    "LazyToolDefinition",
    "LazyToolHeader",
    "LazyToolRuntimeError",
    "discover_lazy_tools",
    "parse_lazy_tool_header",
    "run_lazy_tool",
]
