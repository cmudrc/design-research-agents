"""Contract checks for removed legacy implementation module paths."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.contract
def test_legacy_agent_implementation_module_paths_are_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("design_research_agents.agent.implementations")


@pytest.mark.contract
def test_legacy_agent_internal_module_paths_are_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("design_research_agents.agent.internal")
