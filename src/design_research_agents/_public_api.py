"""Typed public API facade objects grouped by user-facing namespace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from .agent import (
    AgentRuntime,
    MultiStepCodeToolCallingAgent,
    MultiStepJsonToolCallingAgent,
    RuntimeControls,
    SingleStepCodeToolCallingAgent,
    SingleStepDirectLLMAgent,
    SingleStepJsonToolCallingAgent,
    SingleStepRouterAgent,
)
from .contracts import agent as _contracts_agent_module
from .contracts import llm as _contracts_llm_module
from .contracts import orchestrator as _contracts_orchestrator_module
from .contracts import tools as _contracts_tools_module
from .contracts.orchestrator import AgentStep, LogicStep, ToolStep
from .llm import BaseLLMClient, LLMRouter, configure_router_from_yaml
from .llm.backends.default import create_default_llm_client
from .mcp_server import StdioMcpServer, serve_stdio
from .model_selection import (
    HardwareProfile,
    ModelCatalog,
    ModelSelectionConstraints,
    ModelSelectionDecision,
    ModelSelectionIntent,
    ModelSelectionPolicy,
    ModelSelectionPolicyConfig,
)
from .orchestrator import (
    AgentRoutingOrchestrator,
    MixedAgentWorkflowOrchestrator,
    PlanExecuteOrchestrator,
    ProposeAndCritiqueOrchestrator,
    PureToolWorkflowOrchestrator,
    WorkflowRuntime,
)
from .schemas import (
    SCHEMA_NAMES,
    SchemaValidationError,
    load_schema,
    validate_payload_against_schema,
)
from .tools import (
    CoreToolsConfig,
    LazyToolsConfig,
    McpConfig,
    McpServerConfig,
    ToolRuntimeConfig,
    UnifiedToolRuntime,
    load_tool_runtime_config,
)
from .tracing import TraceConfig, configure_tracing


@dataclass(frozen=True, slots=True)
class AgentsNamespace:
    """User-facing namespace for agent runtime implementations."""

    AgentRuntime: type[AgentRuntime] = AgentRuntime
    RuntimeControls: type[RuntimeControls] = RuntimeControls
    SingleStepDirectLLMAgent: type[SingleStepDirectLLMAgent] = SingleStepDirectLLMAgent
    SingleStepRouterAgent: type[SingleStepRouterAgent] = SingleStepRouterAgent
    SingleStepJsonToolCallingAgent: type[SingleStepJsonToolCallingAgent] = (
        SingleStepJsonToolCallingAgent
    )
    SingleStepCodeToolCallingAgent: type[SingleStepCodeToolCallingAgent] = (
        SingleStepCodeToolCallingAgent
    )
    MultiStepCodeToolCallingAgent: type[MultiStepCodeToolCallingAgent] = (
        MultiStepCodeToolCallingAgent
    )
    MultiStepJsonToolCallingAgent: type[MultiStepJsonToolCallingAgent] = (
        MultiStepJsonToolCallingAgent
    )


@dataclass(frozen=True, slots=True)
class WorkflowsNamespace:
    """User-facing namespace for workflow orchestration primitives."""

    WorkflowRuntime: type[WorkflowRuntime] = WorkflowRuntime
    AgentRoutingOrchestrator: type[AgentRoutingOrchestrator] = AgentRoutingOrchestrator
    PlanExecuteOrchestrator: type[PlanExecuteOrchestrator] = PlanExecuteOrchestrator
    ProposeAndCritiqueOrchestrator: type[ProposeAndCritiqueOrchestrator] = (
        ProposeAndCritiqueOrchestrator
    )
    PureToolWorkflowOrchestrator: type[PureToolWorkflowOrchestrator] = PureToolWorkflowOrchestrator
    MixedAgentWorkflowOrchestrator: type[MixedAgentWorkflowOrchestrator] = (
        MixedAgentWorkflowOrchestrator
    )
    AgentStep: type[AgentStep] = AgentStep
    LogicStep: type[LogicStep] = LogicStep
    ToolStep: type[ToolStep] = ToolStep


@dataclass(frozen=True, slots=True)
class LLMNamespace:
    """User-facing namespace for LLM client and router APIs."""

    BaseLLMClient: type[BaseLLMClient] = BaseLLMClient
    LLMRouter: type[LLMRouter] = LLMRouter
    configure_router_from_yaml: Callable[..., object] = configure_router_from_yaml
    create_default_llm_client: Callable[..., object] = create_default_llm_client


@dataclass(frozen=True, slots=True)
class ToolsNamespace:
    """User-facing namespace for tool runtime APIs."""

    UnifiedToolRuntime: type[UnifiedToolRuntime] = UnifiedToolRuntime
    ToolRuntimeConfig: type[ToolRuntimeConfig] = ToolRuntimeConfig
    CoreToolsConfig: type[CoreToolsConfig] = CoreToolsConfig
    LazyToolsConfig: type[LazyToolsConfig] = LazyToolsConfig
    McpConfig: type[McpConfig] = McpConfig
    McpServerConfig: type[McpServerConfig] = McpServerConfig
    load_tool_runtime_config: Callable[..., object] = load_tool_runtime_config


@dataclass(frozen=True, slots=True)
class ModelsNamespace:
    """User-facing namespace for model selection APIs."""

    HardwareProfile: type[HardwareProfile] = HardwareProfile
    ModelCatalog: type[ModelCatalog] = ModelCatalog
    ModelSelectionPolicy: type[ModelSelectionPolicy] = ModelSelectionPolicy
    ModelSelectionIntent: type[ModelSelectionIntent] = ModelSelectionIntent
    ModelSelectionConstraints: type[ModelSelectionConstraints] = ModelSelectionConstraints
    ModelSelectionDecision: type[ModelSelectionDecision] = ModelSelectionDecision
    ModelSelectionPolicyConfig: type[ModelSelectionPolicyConfig] = ModelSelectionPolicyConfig


@dataclass(frozen=True, slots=True)
class TracingNamespace:
    """User-facing namespace for tracing configuration APIs."""

    TraceConfig: type[TraceConfig] = TraceConfig
    configure_tracing: Callable[..., object] = configure_tracing


@dataclass(frozen=True, slots=True)
class ContractsNamespace:
    """User-facing namespace for shared contracts."""

    agent: ModuleType = _contracts_agent_module
    llm: ModuleType = _contracts_llm_module
    tools: ModuleType = _contracts_tools_module
    orchestrator: ModuleType = _contracts_orchestrator_module


@dataclass(frozen=True, slots=True)
class SchemasNamespace:
    """User-facing namespace for packaged JSON schemas."""

    SCHEMA_NAMES: tuple[str, ...] = SCHEMA_NAMES
    SchemaValidationError: type[SchemaValidationError] = SchemaValidationError
    load_schema: Callable[..., object] = load_schema
    validate_payload_against_schema: Callable[..., object] = validate_payload_against_schema


@dataclass(frozen=True, slots=True)
class MCPNamespace:
    """User-facing namespace for built-in MCP runtime support."""

    StdioMcpServer: type[StdioMcpServer] = StdioMcpServer
    serve_stdio: Callable[..., object] = serve_stdio


agents = AgentsNamespace()
workflows = WorkflowsNamespace()
llm = LLMNamespace()
tools = ToolsNamespace()
models = ModelsNamespace()
tracing = TracingNamespace()
contracts = ContractsNamespace()
schemas = SchemasNamespace()
mcp = MCPNamespace()

__all__ = [
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
