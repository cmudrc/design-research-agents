"""Backward-compatible import path for llama-cpp defaults.

Canonical location: ``design_research_agents.llm.backends.default``.
"""

from __future__ import annotations

from design_research_agents.llm.backends.default import create_default_llm_client

__all__ = [
    "create_default_llm_client",
]
