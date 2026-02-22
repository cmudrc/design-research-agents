"""Public API contract tests for curated top-level exports."""

from __future__ import annotations

import importlib

import pytest

import design_research_agents as dra
import design_research_agents.llm as dra_llm
import design_research_agents.memory as dra_memory
import design_research_agents.tools as dra_tools
import design_research_agents.workflow as dra_workflow
from design_research_agents import _contracts as dra_contracts

EXPECTED_PUBLIC_API = [
    "__version__",
    "DirectLLMCall",
    "MultiStepAgent",
    "Toolbox",
    "CallableTool",
    "ScriptTool",
    "McpServer",
    "LogicStep",
    "ToolStep",
    "AgentStep",
    "ModelStep",
    "DelegateBatchStep",
    "LoopStep",
    "MemoryReadStep",
    "MemoryWriteStep",
    "ExecutionResult",
    "LLMRequest",
    "LLMMessage",
    "LLMResponse",
    "ToolResult",
    "Workflow",
    "ConversationPattern",
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
    "VllmServerLLMClient",
    "OllamaLLMClient",
    "SglangServerLLMClient",
    "ModelSelector",
    "Tracer",
]

EXPECTED_TOOLS_API = ["CallableTool", "McpServer", "ScriptTool", "ToolResult", "Toolbox"]


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


def test_top_level_core_result_bundle_exports_resolve() -> None:
    assert dra.ExecutionResult.__name__ == "ExecutionResult"
    assert dra.LLMRequest.__name__ == "LLMRequest"
    assert dra.LLMMessage.__name__ == "LLMMessage"
    assert dra.LLMResponse.__name__ == "LLMResponse"
    assert dra.ToolResult.__name__ == "ToolResult"


def test_contract_symbol_is_available_from_internal_contracts_namespace() -> None:
    assert dra_contracts.LLMMessage.__name__ == "LLMMessage"


def test_llm_module_exports_request_and_message_contracts() -> None:
    assert dra_llm.LLMRequest.__name__ == "LLMRequest"
    assert dra_llm.LLMMessage.__name__ == "LLMMessage"
    assert dra_llm.LLMResponse.__name__ == "LLMResponse"


def test_memory_module_exports_public_memory_facade() -> None:
    assert dra_memory.SQLiteMemoryStore.__name__ == "SQLiteMemoryStore"
    assert dra_memory.LLMEmbeddingProvider.__name__ == "LLMEmbeddingProvider"


def test_workflow_module_exports_execution_result_contract() -> None:
    assert dra_workflow.ExecutionResult.__name__ == "ExecutionResult"


def test_legacy_contracts_namespace_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("design_research_agents.contracts")
