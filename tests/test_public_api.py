"""Public API contract tests for curated top-level exports."""

from __future__ import annotations

import importlib

import pytest

import design_research_agents as dra
import design_research_agents.tools as dra_tools

EXPECTED_PUBLIC_API = [
    "__version__",
    "SingleStepDirectLLMAgent",
    "SingleStepToolRouterAgent",
    "SingleStepJsonToolCallingAgent",
    "SingleStepCodeToolCallingAgent",
    "MultiStepDirectLLMAgent",
    "MultiStepToolRouterAgent",
    "MultiStepJsonToolCallingAgent",
    "MultiStepCodeToolCallingAgent",
    "Toolbox",
    "CallableTool",
    "ScriptTool",
    "McpServer",
    "LogicStep",
    "ToolStep",
    "AgentStep",
    "LoopStep",
    "MemoryReadStep",
    "MemoryWriteStep",
    "Workflow",
    "DebatePattern",
    "PlannerExecutorPattern",
    "ReflexionPattern",
    "RouterPattern",
    "NetworkedPattern",
    "BlackboardPattern",
    "TreeSearchPattern",
    "RagReasoningPattern",
    "LlamaCppServerLLMClient",
    "OpenAIServiceLLMClient",
    "OpenAICompatibleHTTPLLMClient",
    "TransformersLocalLLMClient",
    "MlxLocalLLMClient",
    "ModelSelector",
]

EXPECTED_TOOLS_API = ["CallableTool", "McpServer", "ScriptTool", "Toolbox"]


def test_top_level_exports_match_curated_contract() -> None:
    assert dra.__all__ == EXPECTED_PUBLIC_API


def test_top_level_exports_resolve() -> None:
    for symbol_name in dra.__all__:
        value = getattr(dra, symbol_name)
        assert value is getattr(dra, symbol_name)


def test_tools_module_exports_match_curated_contract() -> None:
    assert dra_tools.__all__ == EXPECTED_TOOLS_API
    for symbol_name in dra_tools.__all__:
        assert getattr(dra_tools, symbol_name) is not None


def test_internal_public_api_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("design_research_agents._public_api")
