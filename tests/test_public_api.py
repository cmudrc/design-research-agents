"""Public API contract tests for strict top-level facade namespaces."""

from __future__ import annotations

import dataclasses

import pytest

import design_research_agents as dra


def test_top_level_exports_only_facade_namespaces_and_version() -> None:
    assert dra.__all__ == [
        "__version__",
        "agents",
        "contracts",
        "llm",
        "mcp",
        "models",
        "schemas",
        "tools",
        "tracing",
        "workflows",
    ]


def test_top_level_legacy_symbols_are_not_available() -> None:
    removed_symbols = (
        "AgentRuntime",
        "AgentStep",
        "BaseLLMClient",
        "SingleStepDirectLLMAgent",
        "HardwareProfile",
        "LLMRouter",
        "LogicStep",
        "ModelCatalog",
        "ModelSelectionConstraints",
        "ModelSelectionDecision",
        "ModelSelectionIntent",
        "ModelSelectionPolicy",
        "ModelSelectionPolicyConfig",
        "MultiStepCodeToolCallingAgent",
        "SingleStepRouterAgent",
        "RuntimeControls",
        "SingleStepCodeToolCallingAgent",
        "SingleStepJsonToolCallingAgent",
        "ToolRuntimeConfig",
        "ToolStep",
        "TraceConfig",
        "UnifiedToolRuntime",
        "WorkflowRuntime",
        "configure_router_from_yaml",
        "configure_tracing",
        "create_default_llm_client",
        "load_tool_runtime_config",
    )
    for symbol_name in removed_symbols:
        assert not hasattr(dra, symbol_name)


def test_facade_namespaces_expose_expected_symbols() -> None:
    expected_namespace_symbols = {
        "agents": (
            "AgentRuntime",
            "RuntimeControls",
            "SingleStepDirectLLMAgent",
            "SingleStepRouterAgent",
            "SingleStepJsonToolCallingAgent",
            "SingleStepCodeToolCallingAgent",
            "MultiStepCodeToolCallingAgent",
        ),
        "workflows": (
            "WorkflowRuntime",
            "AgentRoutingWorkflow",
            "PlanExecuteWorkflow",
            "ProposeAndCritiqueWorkflow",
            "PureToolWorkflow",
            "MixedAgentWorkflow",
            "AgentStep",
            "LogicStep",
            "ToolStep",
        ),
        "llm": (
            "BaseLLMClient",
            "LLMRouter",
            "configure_router_from_yaml",
            "create_default_llm_client",
        ),
        "tools": (
            "UnifiedToolRuntime",
            "ToolRuntimeConfig",
            "load_tool_runtime_config",
        ),
        "models": (
            "HardwareProfile",
            "ModelCatalog",
            "ModelSelectionPolicy",
            "ModelSelectionIntent",
            "ModelSelectionConstraints",
            "ModelSelectionDecision",
            "ModelSelectionPolicyConfig",
        ),
        "tracing": (
            "TraceConfig",
            "configure_tracing",
        ),
        "contracts": (
            "agent",
            "llm",
            "tools",
            "workflow",
        ),
        "schemas": (
            "SCHEMA_NAMES",
            "SchemaValidationError",
            "load_schema",
            "validate_payload_against_schema",
        ),
        "mcp": (
            "StdioMcpServer",
            "serve_stdio",
        ),
    }
    for namespace_name, required_symbols in expected_namespace_symbols.items():
        namespace_value = getattr(dra, namespace_name)
        for symbol_name in required_symbols:
            assert hasattr(namespace_value, symbol_name)


def test_facade_namespaces_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        dra.agents.MultiStepCodeToolCallingAgent = object  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        dra.tools.UnifiedToolRuntime = object  # type: ignore[misc]
