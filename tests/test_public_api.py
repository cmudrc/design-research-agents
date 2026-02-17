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
    "ToolRouterAgent",
    "SingleStepRouterAgent",
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
    "Workflow",
    "DebatePattern",
    "PlannerExecutorPattern",
    "ReflexionPattern",
    "RouterPattern",
    "LlamaCppServerLLMClient",
    "OpenAIServiceLLMClient",
    "OpenAICompatibleHTTPLLMClient",
    "TransformersLocalLLMClient",
    "MlxLocalLLMClient",
    "ModelSelector",
]


def test_top_level_exports_match_curated_contract() -> None:
    assert dra.__all__ == EXPECTED_PUBLIC_API


def test_all_exported_symbols_resolve_and_are_cached() -> None:
    for symbol_name in dra.__all__:
        value = getattr(dra, symbol_name)
        assert value is getattr(dra, symbol_name)


def test_removed_namespace_facades_are_not_available() -> None:
    removed_namespace_facades = (
        "agents",
        "workflows",
        "llm",
        "tools",
        "models",
        "schemas",
        "tracing",
        "contracts",
        "mcp",
    )
    for symbol_name in removed_namespace_facades:
        assert symbol_name not in dra.__all__


def test_agent_runtime_remains_internal() -> None:
    assert "AgentRuntime" not in dra.__all__


def test_workflow_runtime_remains_internal() -> None:
    assert "WorkflowRuntime" not in dra.__all__


def test_tool_config_types_are_not_top_level_exports() -> None:
    hidden_symbols = (
        "ToolRuntimeConfig",
        "CoreToolsConfig",
        "ScriptToolsConfig",
        "McpConfig",
        "load_tool_runtime_config",
    )
    for symbol_name in hidden_symbols:
        assert symbol_name not in dra.__all__


def test_removed_llm_and_schema_symbols_are_not_top_level_exports() -> None:
    hidden_symbols = (
        "BaseLLMClient",
        "LLMRouter",
        "configure_router_from_yaml",
        "resolve_default_model",
        "set_default_router",
        "SCHEMA_NAMES",
        "SchemaValidationError",
        "load_schema",
        "validate_payload_against_schema",
    )
    for symbol_name in hidden_symbols:
        assert symbol_name not in dra.__all__


def test_removed_tracing_and_model_selection_symbols_are_not_top_level_exports() -> None:
    hidden_symbols = (
        "TraceConfig",
        "HardwareProfile",
        "ModelSelectionPolicy",
        "ModelSelectionIntent",
        "ModelSelectionConstraints",
    )
    for symbol_name in hidden_symbols:
        assert symbol_name not in dra.__all__


def test_tools_module_exports_only_unified_runtime() -> None:
    assert dra_tools.__all__ == ["CallableTool", "McpServer", "ScriptTool", "Toolbox"]
    hidden_symbols = (
        "ToolRuntimeConfig",
        "CoreToolsConfig",
        "ScriptToolsConfig",
        "McpConfig",
        "load_tool_runtime_config",
    )
    for symbol_name in hidden_symbols:
        assert not hasattr(dra_tools, symbol_name)


def test_stdio_mcp_server_is_not_top_level_export() -> None:
    assert "StdioMcpServer" not in dra.__all__


def test_internal_public_api_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("design_research_agents._public_api")
