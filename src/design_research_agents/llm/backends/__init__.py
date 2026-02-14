"""Minimal backend package exports.

This namespace intentionally exposes only the default client-construction helper.
"""

from .default import create_default_llm_client

__all__ = [
    "create_default_llm_client",
]
