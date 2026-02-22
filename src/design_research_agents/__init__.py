"""Curated public package interface with deferred top-level exports."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Final

_EXPORTS: Final[dict[str, str]] = {
    "DirectLLMCall": "design_research_agents.agent:DirectLLMCall",
    "MultiStepAgent": "design_research_agents.agent:MultiStepAgent",
    "Toolbox": "design_research_agents.tools:Toolbox",
    "CallableTool": "design_research_agents.tools:CallableTool",
    "ScriptTool": "design_research_agents.tools:ScriptTool",
    "McpServer": "design_research_agents.tools:McpServer",
    "LogicStep": "design_research_agents._contracts:LogicStep",
    "ToolStep": "design_research_agents._contracts:ToolStep",
    "AgentStep": "design_research_agents._contracts:AgentStep",
    "ModelStep": "design_research_agents._contracts:ModelStep",
    "DelegateBatchStep": "design_research_agents._contracts:DelegateBatchStep",
    "LoopStep": "design_research_agents._contracts:LoopStep",
    "MemoryReadStep": "design_research_agents._contracts:MemoryReadStep",
    "MemoryWriteStep": "design_research_agents._contracts:MemoryWriteStep",
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
    "VllmServerLLMClient": "design_research_agents.llm:VllmServerLLMClient",
    "OllamaLLMClient": "design_research_agents.llm:OllamaLLMClient",
    "SglangServerLLMClient": "design_research_agents.llm:SglangServerLLMClient",
    "ModelSelector": "design_research_agents._model_selection:ModelSelector",
    "Tracer": "design_research_agents._tracing:Tracer",
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
    from ._contracts import AgentStep as AgentStep
    from ._contracts import DelegateBatchStep as DelegateBatchStep
    from ._contracts import LogicStep as LogicStep
    from ._contracts import LoopStep as LoopStep
    from ._contracts import MemoryReadStep as MemoryReadStep
    from ._contracts import MemoryWriteStep as MemoryWriteStep
    from ._contracts import ModelStep as ModelStep
    from ._contracts import ToolStep as ToolStep
    from ._model_selection import ModelSelector as ModelSelector
    from ._tracing import Tracer as Tracer
    from .agent import DirectLLMCall as DirectLLMCall
    from .agent import MultiStepAgent as MultiStepAgent
    from .llm import LlamaCppServerLLMClient as LlamaCppServerLLMClient
    from .llm import MlxLocalLLMClient as MlxLocalLLMClient
    from .llm import OllamaLLMClient as OllamaLLMClient
    from .llm import OpenAICompatibleHTTPLLMClient as OpenAICompatibleHTTPLLMClient
    from .llm import OpenAIServiceLLMClient as OpenAIServiceLLMClient
    from .llm import SglangServerLLMClient as SglangServerLLMClient
    from .llm import TransformersLocalLLMClient as TransformersLocalLLMClient
    from .llm import VllmServerLLMClient as VllmServerLLMClient
    from .tools import CallableTool as CallableTool
    from .tools import McpServer as McpServer
    from .tools import ScriptTool as ScriptTool
    from .tools import Toolbox as Toolbox
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
