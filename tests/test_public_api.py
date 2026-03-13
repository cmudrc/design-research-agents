"""Public API contract tests for curated top-level exports."""

from __future__ import annotations

import importlib
import inspect

import pytest

import design_research_agents as dra
import design_research_agents.llm as dra_llm
import design_research_agents.memory as dra_memory
import design_research_agents.patterns as dra_patterns
import design_research_agents.tools as dra_tools
import design_research_agents.workflow as dra_workflow
from design_research_agents import _contracts as dra_contracts
from design_research_agents._contracts._tools import ToolRuntime

EXPECTED_PUBLIC_API = [
    "__version__",
    "DirectLLMCall",
    "MultiStepAgent",
    "Toolbox",
    "CallableToolConfig",
    "ScriptToolConfig",
    "MCPServerConfig",
    "LogicStep",
    "ToolStep",
    "DelegateStep",
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
    "CompiledExecution",
    "TwoSpeakerConversationPattern",
    "DebatePattern",
    "PlanExecutePattern",
    "ProposeCriticPattern",
    "RalphLoopPattern",
    "NominalTeamPattern",
    "RouterDelegatePattern",
    "RoundBasedCoordinationPattern",
    "BlackboardPattern",
    "TreeSearchPattern",
    "RAGPattern",
    "AnthropicServiceLLMClient",
    "AzureOpenAIServiceLLMClient",
    "GeminiServiceLLMClient",
    "GroqServiceLLMClient",
    "LlamaCppServerLLMClient",
    "OpenAIServiceLLMClient",
    "OpenAICompatibleHTTPLLMClient",
    "TransformersLocalLLMClient",
    "MLXLocalLLMClient",
    "VLLMServerLLMClient",
    "OllamaLLMClient",
    "SGLangServerLLMClient",
    "ModelSelector",
    "Tracer",
]

EXPECTED_TOOLS_API = ["CallableToolConfig", "MCPServerConfig", "ScriptToolConfig", "ToolResult", "Toolbox"]
EXPECTED_WORKFLOW_API = [
    "CompiledExecution",
    "DelegateBatchCall",
    "DelegateBatchStep",
    "DelegateStep",
    "ExecutionResult",
    "LogicStep",
    "LoopStep",
    "MemoryReadStep",
    "MemoryWriteStep",
    "ModelStep",
    "ToolStep",
    "Workflow",
    "WorkflowArtifact",
    "WorkflowArtifactSource",
    "list_of",
    "scalar",
    "typed_dict",
]
EXPECTED_PATTERNS_API = [
    "BlackboardPattern",
    "TwoSpeakerConversationPattern",
    "DebatePattern",
    "RoundBasedCoordinationPattern",
    "PlanExecutePattern",
    "RAGPattern",
    "ProposeCriticPattern",
    "RalphLoopPattern",
    "NominalTeamPattern",
    "RouterDelegatePattern",
    "TreeSearchPattern",
]
LEGACY_PUBLIC_SYMBOLS = [
    "CallableTool",
    "ScriptTool",
    "McpServer",
    "PlannerExecutorPattern",
    "RouterPattern",
    "RagReasoningPattern",
    "ConversationPattern",
    "ReflexionPattern",
    "AgentRoutingPattern",
    "NetworkedPattern",
    "BeamSearchPattern",
    "MlxLocalLLMClient",
    "VllmServerLLMClient",
    "SglangServerLLMClient",
]


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


def test_workflow_module_exports_builder_surface_contract() -> None:
    assert dra_workflow.__all__ == EXPECTED_WORKFLOW_API
    for symbol_name in dra_workflow.__all__:
        assert getattr(dra_workflow, symbol_name) is not None


def test_patterns_module_exports_match_curated_contract() -> None:
    assert dra_patterns.__all__ == EXPECTED_PATTERNS_API
    for symbol_name in dra_patterns.__all__:
        assert getattr(dra_patterns, symbol_name) is not None


def test_legacy_public_symbols_are_absent_from_public_facades() -> None:
    for symbol_name in LEGACY_PUBLIC_SYMBOLS:
        with pytest.raises(AttributeError):
            getattr(dra, symbol_name)
        with pytest.raises(AttributeError):
            getattr(dra_patterns, symbol_name)
        with pytest.raises(AttributeError):
            getattr(dra_llm, symbol_name)
        with pytest.raises(AttributeError):
            getattr(dra_tools, symbol_name)


def test_public_entrypoint_signatures_use_prompt_or_input_only() -> None:
    workflow_run_params = inspect.signature(dra_workflow.Workflow.run).parameters
    assert "input" in workflow_run_params
    assert "input_data" not in workflow_run_params

    tool_runtime_invoke_params = inspect.signature(ToolRuntime.invoke).parameters
    assert "input" in tool_runtime_invoke_params
    assert "input_dict" not in tool_runtime_invoke_params

    toolbox_invoke_params = inspect.signature(dra_tools.Toolbox.invoke).parameters
    assert "input" in toolbox_invoke_params
    assert "input_dict" not in toolbox_invoke_params

    toolbox_invoke_dict_params = inspect.signature(dra_tools.Toolbox.invoke_dict).parameters
    assert "input" in toolbox_invoke_dict_params
    assert "input_dict" not in toolbox_invoke_dict_params

    for pattern_name in EXPECTED_PATTERNS_API:
        pattern_cls = getattr(dra_patterns, pattern_name)
        pattern_run_params = inspect.signature(pattern_cls.run).parameters
        assert "prompt" in pattern_run_params
        assert "input" not in pattern_run_params
        assert "input_data" not in pattern_run_params


def test_workflow_module_does_not_export_patterns() -> None:
    for symbol_name in EXPECTED_PATTERNS_API:
        with pytest.raises(AttributeError):
            getattr(dra_workflow, symbol_name)


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


def test_legacy_contracts_namespace_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("design_research_agents.contracts")
