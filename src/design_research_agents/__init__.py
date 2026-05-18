"""Curated public package interface with lazily resolved top-level exports."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Final

from design_research_agents._lazy_exports import module_dir, resolve_lazy_export

_EXPORTS: Final[dict[str, str]] = {
    "DirectLLMCall": "design_research_agents.agent:DirectLLMCall",
    "MultiStepAgent": "design_research_agents.agent:MultiStepAgent",
    "SkillsConfig": "design_research_agents.skills:SkillsConfig",
    "SeededRandomBaselineAgent": "design_research_agents.agent:SeededRandomBaselineAgent",
    "PromptWorkflowAgent": "design_research_agents.agent:PromptWorkflowAgent",
    "Toolbox": "design_research_agents.tools:Toolbox",
    "CallableToolConfig": "design_research_agents.tools:CallableToolConfig",
    "ScriptToolConfig": "design_research_agents.tools:ScriptToolConfig",
    "MCPServerConfig": "design_research_agents.tools:MCPServerConfig",
    "LogicStep": "design_research_agents._contracts:LogicStep",
    "ToolStep": "design_research_agents._contracts:ToolStep",
    "DelegateStep": "design_research_agents._contracts:DelegateStep",
    "ModelStep": "design_research_agents._contracts:ModelStep",
    "DelegateBatchStep": "design_research_agents._contracts:DelegateBatchStep",
    "LoopStep": "design_research_agents._contracts:LoopStep",
    "MemoryReadStep": "design_research_agents._contracts:MemoryReadStep",
    "MemoryWriteStep": "design_research_agents._contracts:MemoryWriteStep",
    "ExecutionResult": "design_research_agents._contracts:ExecutionResult",
    "LLMRequest": "design_research_agents.llm:LLMRequest",
    "LLMMessage": "design_research_agents.llm:LLMMessage",
    "LLMResponse": "design_research_agents.llm:LLMResponse",
    "ToolResult": "design_research_agents.tools:ToolResult",
    "Workflow": "design_research_agents.workflow:Workflow",
    "CompiledExecution": "design_research_agents.workflow:CompiledExecution",
    "build_json_prompt_workflow": "design_research_agents.workflow:build_json_prompt_workflow",
    "TwoSpeakerConversationPattern": "design_research_agents.patterns:TwoSpeakerConversationPattern",
    "DebatePattern": "design_research_agents.patterns:DebatePattern",
    "PlanExecutePattern": "design_research_agents.patterns:PlanExecutePattern",
    "ProposeCriticPattern": "design_research_agents.patterns:ProposeCriticPattern",
    "RalphLoopPattern": "design_research_agents.patterns:RalphLoopPattern",
    "NominalTeamPattern": "design_research_agents.patterns:NominalTeamPattern",
    "RouterDelegatePattern": "design_research_agents.patterns:RouterDelegatePattern",
    "RoundBasedCoordinationPattern": "design_research_agents.patterns:RoundBasedCoordinationPattern",
    "BlackboardPattern": "design_research_agents.patterns:BlackboardPattern",
    "TreeSearchPattern": "design_research_agents.patterns:TreeSearchPattern",
    "RAGPattern": "design_research_agents.patterns:RAGPattern",
    "SimulatedAnnealingPattern": "design_research_agents.patterns:SimulatedAnnealingPattern",
    "AnthropicServiceLLMClient": "design_research_agents.llm:AnthropicServiceLLMClient",
    "AzureOpenAIServiceLLMClient": "design_research_agents.llm:AzureOpenAIServiceLLMClient",
    "GeminiServiceLLMClient": "design_research_agents.llm:GeminiServiceLLMClient",
    "GroqServiceLLMClient": "design_research_agents.llm:GroqServiceLLMClient",
    "LlamaCppServerLLMClient": "design_research_agents.llm:LlamaCppServerLLMClient",
    "OpenAIServiceLLMClient": "design_research_agents.llm:OpenAIServiceLLMClient",
    "OpenAICompatibleHTTPLLMClient": "design_research_agents.llm:OpenAICompatibleHTTPLLMClient",
    "TransformersLocalLLMClient": "design_research_agents.llm:TransformersLocalLLMClient",
    "MLXLocalLLMClient": "design_research_agents.llm:MLXLocalLLMClient",
    "VLLMServerLLMClient": "design_research_agents.llm:VLLMServerLLMClient",
    "OllamaLLMClient": "design_research_agents.llm:OllamaLLMClient",
    "SGLangServerLLMClient": "design_research_agents.llm:SGLangServerLLMClient",
    "ModelCatalog": "design_research_agents._model_selection:ModelCatalog",
    "ModelFlight": "design_research_agents._model_selection:ModelFlight",
    "ModelFlightRegistry": "design_research_agents._model_selection:ModelFlightRegistry",
    "ModelSelector": "design_research_agents._model_selection:ModelSelector",
    "Tracer": "design_research_agents._tracing:Tracer",
}

__all__ = ["__version__", *_EXPORTS.keys()]
__all__.append("integration")

try:
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    __version__ = "0+unknown"


def __getattr__(name: str) -> object:
    """Resolve and cache one deferred public export.

    Args:
        name: Public symbol name requested from the package module.

    Returns:
        Resolved export object.

    Raises:
        AttributeError: If ``name`` is not part of the public export map.
    """
    if name == "integration":
        module = import_module("design_research_agents.integration")
        globals()[name] = module
        return module

    return resolve_lazy_export(
        module_name=__name__,
        exports=_EXPORTS,
        export_name=name,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    """Return package attributes, including deferred exports.

    Returns:
        Sorted attribute list for interactive discovery.
    """
    return module_dir(globals(), __all__)


if TYPE_CHECKING:
    from ._contracts import DelegateBatchStep as DelegateBatchStep
    from ._contracts import DelegateStep as DelegateStep
    from ._contracts import ExecutionResult as ExecutionResult
    from ._contracts import LLMMessage as LLMMessage
    from ._contracts import LLMRequest as LLMRequest
    from ._contracts import LLMResponse as LLMResponse
    from ._contracts import LogicStep as LogicStep
    from ._contracts import LoopStep as LoopStep
    from ._contracts import MemoryReadStep as MemoryReadStep
    from ._contracts import MemoryWriteStep as MemoryWriteStep
    from ._contracts import ModelStep as ModelStep
    from ._contracts import ToolStep as ToolStep
    from ._model_selection import ModelCatalog as ModelCatalog
    from ._model_selection import ModelFlight as ModelFlight
    from ._model_selection import ModelFlightRegistry as ModelFlightRegistry
    from ._model_selection import ModelSelector as ModelSelector
    from ._tracing import Tracer as Tracer
    from .agent import DirectLLMCall as DirectLLMCall
    from .agent import MultiStepAgent as MultiStepAgent
    from .agent import PromptWorkflowAgent as PromptWorkflowAgent
    from .agent import SeededRandomBaselineAgent as SeededRandomBaselineAgent
    from .integration import execute_agent_run as execute_agent_run
    from .llm import AnthropicServiceLLMClient as AnthropicServiceLLMClient
    from .llm import AzureOpenAIServiceLLMClient as AzureOpenAIServiceLLMClient
    from .llm import GeminiServiceLLMClient as GeminiServiceLLMClient
    from .llm import GroqServiceLLMClient as GroqServiceLLMClient
    from .llm import LlamaCppServerLLMClient as LlamaCppServerLLMClient
    from .llm import MLXLocalLLMClient as MLXLocalLLMClient
    from .llm import OllamaLLMClient as OllamaLLMClient
    from .llm import OpenAICompatibleHTTPLLMClient as OpenAICompatibleHTTPLLMClient
    from .llm import OpenAIServiceLLMClient as OpenAIServiceLLMClient
    from .llm import SGLangServerLLMClient as SGLangServerLLMClient
    from .llm import TransformersLocalLLMClient as TransformersLocalLLMClient
    from .llm import VLLMServerLLMClient as VLLMServerLLMClient
    from .patterns import BlackboardPattern as BlackboardPattern
    from .patterns import DebatePattern as DebatePattern
    from .patterns import NominalTeamPattern as NominalTeamPattern
    from .patterns import PlanExecutePattern as PlanExecutePattern
    from .patterns import ProposeCriticPattern as ProposeCriticPattern
    from .patterns import RAGPattern as RAGPattern
    from .patterns import RalphLoopPattern as RalphLoopPattern
    from .patterns import RoundBasedCoordinationPattern as RoundBasedCoordinationPattern
    from .patterns import RouterDelegatePattern as RouterDelegatePattern
    from .patterns import SimulatedAnnealingPattern as SimulatedAnnealingPattern
    from .patterns import TreeSearchPattern as TreeSearchPattern
    from .patterns import TwoSpeakerConversationPattern as TwoSpeakerConversationPattern
    from .skills import SkillsConfig as SkillsConfig
    from .tools import CallableToolConfig as CallableToolConfig
    from .tools import MCPServerConfig as MCPServerConfig
    from .tools import ScriptToolConfig as ScriptToolConfig
    from .tools import Toolbox as Toolbox
    from .tools import ToolResult as ToolResult
    from .workflow import CompiledExecution as CompiledExecution
    from .workflow import Workflow as Workflow
    from .workflow import build_json_prompt_workflow as build_json_prompt_workflow
