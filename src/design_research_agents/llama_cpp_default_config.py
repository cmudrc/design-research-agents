"""Backward-compatible import path for llama-cpp defaults.

Canonical location: ``design_research_agents.llm.backends.default``.
"""

from __future__ import annotations

from design_research_agents.llm.backends.default import (
    DEFAULT_LLAMA_CPP_SETTINGS,
    DefaultLlamaCppSettings,
    configure_default_llama_cpp_backend,
    create_default_llm_client,
)

__all__ = [
    "DEFAULT_LLAMA_CPP_SETTINGS",
    "DefaultLlamaCppSettings",
    "configure_default_llama_cpp_backend",
    "create_default_llm_client",
]
