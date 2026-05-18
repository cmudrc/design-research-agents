"""Stable public exports for building and running workflows."""

from __future__ import annotations

from design_research_agents._contracts import (
    DelegateBatchCall,
    DelegateBatchStep,
    DelegateStep,
    ExecutionResult,
    LogicStep,
    LoopStep,
    MemoryReadStep,
    MemoryWriteStep,
    ModelStep,
    ToolStep,
    WorkflowArtifact,
    WorkflowArtifactSource,
)

from ._compiled import CompiledExecution
from ._json_prompt import build_json_prompt_workflow
from ._schema_helpers import list_of, scalar, typed_dict
from .workflow import Workflow

__all__ = [
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
    "build_json_prompt_workflow",
    "list_of",
    "scalar",
    "typed_dict",
]


def __dir__() -> list[str]:
    """Return workflow module attributes, including re-exported helpers.

    Returns:
        Sorted attribute names visible on this module.
    """
    return sorted(set(globals()) | set(__all__))
