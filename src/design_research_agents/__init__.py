"""Curated public package interface with deferred top-level exports."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Final

_EXPORTS: Final[dict[str, str]] = {
    "SingleStepDirectLLMAgent": "design_research_agents.agent:SingleStepDirectLLMAgent",
    "SingleStepToolRouterAgent": "design_research_agents.agent:SingleStepToolRouterAgent",
    "SingleStepJsonToolCallingAgent": "design_research_agents.agent:SingleStepJsonToolCallingAgent",
    "SingleStepCodeToolCallingAgent": "design_research_agents.agent:SingleStepCodeToolCallingAgent",
    "MultiStepDirectLLMAgent": "design_research_agents.agent:MultiStepDirectLLMAgent",
    "MultiStepToolRouterAgent": "design_research_agents.agent:MultiStepToolRouterAgent",
    "MultiStepJsonToolCallingAgent": "design_research_agents.agent:MultiStepJsonToolCallingAgent",
    "MultiStepCodeToolCallingAgent": "design_research_agents.agent:MultiStepCodeToolCallingAgent",
    "Toolbox": "design_research_agents.tools:Toolbox",
    "CallableTool": "design_research_agents.tools:CallableTool",
    "ScriptTool": "design_research_agents.tools:ScriptTool",
    "McpServer": "design_research_agents.tools:McpServer",
    "LogicStep": "design_research_agents.contracts:LogicStep",
    "ToolStep": "design_research_agents.contracts:ToolStep",
    "AgentStep": "design_research_agents.contracts:AgentStep",
    "LoopStep": "design_research_agents.contracts:LoopStep",
    "MemoryReadStep": "design_research_agents.contracts:MemoryReadStep",
    "MemoryWriteStep": "design_research_agents.contracts:MemoryWriteStep",
    "Workflow": "design_research_agents.workflow:Workflow",
    "ConversationPattern": "design_research_agents.workflow:ConversationPattern",
    "DebatePattern": "design_research_agents.workflow:DebatePattern",
    "PlannerExecutorPattern": "design_research_agents.workflow:PlannerExecutorPattern",
    "ReflexionPattern": "design_research_agents.workflow:ReflexionPattern",
    "RouterPattern": "design_research_agents.workflow:RouterPattern",
    "NetworkedPattern": "design_research_agents.workflow:NetworkedPattern",
    "BlackboardPattern": "design_research_agents.workflow:BlackboardPattern",
    "TreeSearchPattern": "design_research_agents.workflow:TreeSearchPattern",
    "RagReasoningPattern": "design_research_agents.workflow:RagReasoningPattern",
    "LlamaCppServerLLMClient": "design_research_agents.llm:LlamaCppServerLLMClient",
    "OpenAIServiceLLMClient": "design_research_agents.llm:OpenAIServiceLLMClient",
    "OpenAICompatibleHTTPLLMClient": "design_research_agents.llm:OpenAICompatibleHTTPLLMClient",
    "TransformersLocalLLMClient": "design_research_agents.llm:TransformersLocalLLMClient",
    "MlxLocalLLMClient": "design_research_agents.llm:MlxLocalLLMClient",
    "ModelSelector": "design_research_agents.model_selection:ModelSelector",
    "Agent": "design_research_agents.contracts:Agent",
    "ExecutionResult": "design_research_agents.contracts:ExecutionResult",
    "LLMClient": "design_research_agents.contracts:LLMClient",
    "LLMMessage": "design_research_agents.contracts:LLMMessage",
    "LLMRole": "design_research_agents.contracts.llm:LLMRole",
    "LLMChatParams": "design_research_agents.contracts:LLMChatParams",
    "LLMRequest": "design_research_agents.contracts:LLMRequest",
    "TaskProfile": "design_research_agents.contracts:TaskProfile",
    "ToolRuntime": "design_research_agents.contracts:ToolRuntime",
    "ToolSpec": "design_research_agents.contracts:ToolSpec",
    "ToolMetadata": "design_research_agents.contracts:ToolMetadata",
    "ToolSideEffects": "design_research_agents.contracts:ToolSideEffects",
    "ToolCostHints": "design_research_agents.contracts:ToolCostHints",
    "MemoryStore": "design_research_agents.contracts:MemoryStore",
    "MemorySearchQuery": "design_research_agents.contracts:MemorySearchQuery",
    "MemoryWriteRecord": "design_research_agents.contracts:MemoryWriteRecord",
    "WorkflowDelegate": "design_research_agents.contracts.workflow:WorkflowDelegate",
    "WorkflowDelegateRunner": "design_research_agents.contracts:WorkflowDelegateRunner",
    "WorkflowExecutionMode": "design_research_agents.contracts:WorkflowExecutionMode",
    "WorkflowFailurePolicy": "design_research_agents.contracts:WorkflowFailurePolicy",
    "WorkflowStep": "design_research_agents.contracts:WorkflowStep",
    "WorkflowInputMode": "design_research_agents.workflow.workflow:WorkflowInputMode",
    "WorkflowArtifact": "design_research_agents.contracts:WorkflowArtifact",
    "WorkflowArtifactSource": "design_research_agents.contracts:WorkflowArtifactSource",
    "WorkflowArtifactsBuilder": "design_research_agents.contracts:WorkflowArtifactsBuilder",
    "ToolStepInputBuilder": "design_research_agents.contracts:ToolStepInputBuilder",
    "AgentStepPromptBuilder": "design_research_agents.contracts:AgentStepPromptBuilder",
    "LogicStepHandler": "design_research_agents.contracts:LogicStepHandler",
    "MemoryReadQueryBuilder": "design_research_agents.contracts:MemoryReadQueryBuilder",
    "MemoryWriteRecordsBuilder": "design_research_agents.contracts:MemoryWriteRecordsBuilder",
    "LoopStepContinuePredicate": "design_research_agents.contracts:LoopStepContinuePredicate",
    "LoopStepStateReducer": "design_research_agents.contracts:LoopStepStateReducer",
    "LoopStepTerminationReason": "design_research_agents.contracts:LoopStepTerminationReason",
    "Tracer": "design_research_agents.tracing:Tracer",
}

__all__ = ["__version__", *_EXPORTS.keys()]

try:
    __version__ = version("design-research-agents")
    """The current version of the design-research-agents package."""
except PackageNotFoundError:
    __version__ = "unknown"
    """The current version of the design-research-agents package."""


def __getattr__(name: str) -> object:
    """Lazily resolve and cache one public export.

    Args:
        name: Public symbol name requested from the package module.

    Returns:
        Resolved export object.

    Raises:
        AttributeError: If ``name`` is not part of the public export map.
    """
    export_ref = _EXPORTS.get(name)
    if export_ref is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = export_ref.split(":")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package attribute names including deferred exports.

    Returns:
        Sorted attribute list for interactive discovery.
    """
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .agent import MultiStepCodeToolCallingAgent as MultiStepCodeToolCallingAgent
    from .agent import MultiStepDirectLLMAgent as MultiStepDirectLLMAgent
    from .agent import MultiStepJsonToolCallingAgent as MultiStepJsonToolCallingAgent
    from .agent import MultiStepToolRouterAgent as MultiStepToolRouterAgent
    from .agent import SingleStepCodeToolCallingAgent as SingleStepCodeToolCallingAgent
    from .agent import SingleStepDirectLLMAgent as SingleStepDirectLLMAgent
    from .agent import SingleStepJsonToolCallingAgent as SingleStepJsonToolCallingAgent
    from .agent import SingleStepToolRouterAgent as SingleStepToolRouterAgent
    from .contracts import Agent as Agent
    from .contracts import AgentStep as AgentStep
    from .contracts import AgentStepPromptBuilder as AgentStepPromptBuilder
    from .contracts import ExecutionResult as ExecutionResult
    from .contracts import LLMChatParams as LLMChatParams
    from .contracts import LLMClient as LLMClient
    from .contracts import LLMMessage as LLMMessage
    from .contracts import LogicStep as LogicStep
    from .contracts import LogicStepHandler as LogicStepHandler
    from .contracts import LoopStep as LoopStep
    from .contracts import LoopStepContinuePredicate as LoopStepContinuePredicate
    from .contracts import LoopStepStateReducer as LoopStepStateReducer
    from .contracts import LoopStepTerminationReason as LoopStepTerminationReason
    from .contracts import MemoryReadQueryBuilder as MemoryReadQueryBuilder
    from .contracts import MemoryReadStep as MemoryReadStep
    from .contracts import MemorySearchQuery as MemorySearchQuery
    from .contracts import MemoryStore as MemoryStore
    from .contracts import MemoryWriteRecord as MemoryWriteRecord
    from .contracts import MemoryWriteRecordsBuilder as MemoryWriteRecordsBuilder
    from .contracts import MemoryWriteStep as MemoryWriteStep
    from .contracts import TaskProfile as TaskProfile
    from .contracts import ToolCostHints as ToolCostHints
    from .contracts import ToolMetadata as ToolMetadata
    from .contracts import ToolRuntime as ToolRuntime
    from .contracts import ToolSideEffects as ToolSideEffects
    from .contracts import ToolSpec as ToolSpec
    from .contracts import ToolStep as ToolStep
    from .contracts import ToolStepInputBuilder as ToolStepInputBuilder
    from .contracts import WorkflowArtifact as WorkflowArtifact
    from .contracts import WorkflowArtifactsBuilder as WorkflowArtifactsBuilder
    from .contracts import WorkflowArtifactSource as WorkflowArtifactSource
    from .contracts import WorkflowDelegateRunner as WorkflowDelegateRunner
    from .contracts import WorkflowExecutionMode as WorkflowExecutionMode
    from .contracts import WorkflowFailurePolicy as WorkflowFailurePolicy
    from .contracts import WorkflowStep as WorkflowStep
    from .contracts.llm import LLMRequest as LLMRequest
    from .contracts.llm import LLMRole as LLMRole
    from .contracts.workflow import WorkflowDelegate as WorkflowDelegate
    from .llm import LlamaCppServerLLMClient as LlamaCppServerLLMClient
    from .llm import MlxLocalLLMClient as MlxLocalLLMClient
    from .llm import OpenAICompatibleHTTPLLMClient as OpenAICompatibleHTTPLLMClient
    from .llm import OpenAIServiceLLMClient as OpenAIServiceLLMClient
    from .llm import TransformersLocalLLMClient as TransformersLocalLLMClient
    from .model_selection import ModelSelector as ModelSelector
    from .tools import CallableTool as CallableTool
    from .tools import McpServer as McpServer
    from .tools import ScriptTool as ScriptTool
    from .tools import Toolbox as Toolbox
    from .tracing import Tracer as Tracer
    from .workflow import BlackboardPattern as BlackboardPattern
    from .workflow import ConversationPattern as ConversationPattern
    from .workflow import DebatePattern as DebatePattern
    from .workflow import NetworkedPattern as NetworkedPattern
    from .workflow import PlannerExecutorPattern as PlannerExecutorPattern
    from .workflow import RagReasoningPattern as RagReasoningPattern
    from .workflow import ReflexionPattern as ReflexionPattern
    from .workflow import RouterPattern as RouterPattern
    from .workflow import TreeSearchPattern as TreeSearchPattern
    from .workflow import Workflow as Workflow
    from .workflow.workflow import WorkflowInputMode as WorkflowInputMode
