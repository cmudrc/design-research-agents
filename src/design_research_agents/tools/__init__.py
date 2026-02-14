"""Tool runtime implementations and default tool specification helpers.

The exported runtime is intentionally lightweight and in-memory so examples and
tests can run without external infrastructure. The helper constructors expose
default tool specifications used by agents and integration tests.
"""

from .base_runtime import BaseToolRuntime

__all__ = ["BaseToolRuntime"]
