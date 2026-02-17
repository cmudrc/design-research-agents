"""Curated public package interface with deferred top-level exports."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Final

_EXPORTS: Final[dict[str, str]] = {
    "SingleStepDirectLLMAgent": "design_research_agents.agent:SingleStepDirectLLMAgent",
    "SingleStepRouterAgent": "design_research_agents.agent:SingleStepRouterAgent",
    "SingleStepJsonToolCallingAgent": "design_research_agents.agent:SingleStepJsonToolCallingAgent",
    "SingleStepCodeToolCallingAgent": "design_research_agents.agent:SingleStepCodeToolCallingAgent",
    "MultiStepJsonToolCallingAgent": "design_research_agents.agent:MultiStepJsonToolCallingAgent",
    "MultiStepCodeToolCallingAgent": "design_research_agents.agent:MultiStepCodeToolCallingAgent",
    "Toolbox": "design_research_agents.tools:Toolbox",
    "CallableTool": "design_research_agents.tools:CallableTool",
    "ScriptTool": "design_research_agents.tools:ScriptTool",
    "McpServer": "design_research_agents.tools:McpServer",
    "PlanExecuteWorkflow": "design_research_agents.workflow:PlanExecuteWorkflow",
    "ProposeAndCritiqueWorkflow": "design_research_agents.workflow:ProposeAndCritiqueWorkflow",
    "AgentRoutingWorkflow": "design_research_agents.workflow:AgentRoutingWorkflow",
    "DebatePattern": "design_research_agents.workflow:DebatePattern",
    "PureToolWorkflow": "design_research_agents.workflow:PureToolWorkflow",
    "MixedAgentWorkflow": "design_research_agents.workflow:MixedAgentWorkflow",
    "LlamaCppServerLLMClient": "design_research_agents.llm:LlamaCppServerLLMClient",
    "OpenAIServiceLLMClient": "design_research_agents.llm:OpenAIServiceLLMClient",
    "OpenAICompatibleHTTPLLMClient": "design_research_agents.llm:OpenAICompatibleHTTPLLMClient",
    "TransformersLocalLLMClient": "design_research_agents.llm:TransformersLocalLLMClient",
    "MlxLocalLLMClient": "design_research_agents.llm:MlxLocalLLMClient",
    "ModelSelector": "design_research_agents.model_selection:ModelSelector",
}

__all__ = ["__version__", *_EXPORTS.keys()]

try:
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    __version__ = "unknown"


def __getattr__(name: str) -> object:
    export_ref = _EXPORTS.get(name)
    if export_ref is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = export_ref.split(":")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .agent import MultiStepCodeToolCallingAgent as MultiStepCodeToolCallingAgent
    from .agent import MultiStepJsonToolCallingAgent as MultiStepJsonToolCallingAgent
    from .agent import SingleStepCodeToolCallingAgent as SingleStepCodeToolCallingAgent
    from .agent import SingleStepDirectLLMAgent as SingleStepDirectLLMAgent
    from .agent import SingleStepJsonToolCallingAgent as SingleStepJsonToolCallingAgent
    from .agent import SingleStepRouterAgent as SingleStepRouterAgent
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
    from .workflow import AgentRoutingWorkflow as AgentRoutingWorkflow
    from .workflow import DebatePattern as DebatePattern
    from .workflow import MixedAgentWorkflow as MixedAgentWorkflow
    from .workflow import PlanExecuteWorkflow as PlanExecuteWorkflow
    from .workflow import ProposeAndCritiqueWorkflow as ProposeAndCritiqueWorkflow
    from .workflow import PureToolWorkflow as PureToolWorkflow
