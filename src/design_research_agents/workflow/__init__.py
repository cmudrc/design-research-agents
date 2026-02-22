"""Public workflow builder exports."""

from __future__ import annotations

from design_research_agents._contracts import (
    AgentStep,
    DelegateBatchCall,
    DelegateBatchStep,
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

from ._schema_helpers import list_of, scalar, typed_dict
from .workflow import Workflow

__all__ = [
    "AgentStep",
    "DelegateBatchCall",
    "DelegateBatchStep",
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


def __dir__() -> list[str]:
    """Return workflow module attributes.

    Returns:
        Sorted attribute names visible on this module.
    """
    return sorted(set(globals()) | set(__all__))
