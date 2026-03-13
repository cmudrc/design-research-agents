"""Shared workflow-diagram helpers used by Mermaid and SVG renderers."""

from __future__ import annotations

from html import escape as html_escape

from design_research_agents._contracts._workflow import (
    DelegateBatchStep,
    DelegateStep,
    LogicStep,
    LoopStep,
    MemoryReadStep,
    MemoryWriteStep,
    ModelStep,
    ToolStep,
    WorkflowStep,
)

_ALLOWED_DIRECTIONS = frozenset({"TD", "LR", "BT", "RL"})


def _normalize_direction(direction: str) -> str:
    """Normalize and validate one diagram direction token."""
    normalized_direction = direction.strip().upper()
    if normalized_direction not in _ALLOWED_DIRECTIONS:
        raise ValueError("direction must be one of: " + ", ".join(sorted(_ALLOWED_DIRECTIONS)))
    return normalized_direction


def _qualified_step_id(prefix: tuple[str, ...], step_id: str) -> str:
    """Return one step id qualified by its parent loop path when present."""
    if not prefix:
        return step_id
    return "::".join((*prefix, step_id))


def _step_label(step: WorkflowStep, qualified_step_id: str) -> str:
    """Build one stable label for a workflow step."""
    step_kind = type(step).__name__
    detail = _step_detail(step)
    if detail is None:
        return f"{qualified_step_id}\n{step_kind}"
    return f"{qualified_step_id}\n{step_kind}\n{detail}"


def _step_detail(step: WorkflowStep) -> str | None:
    """Return one compact step-detail string when available."""
    if isinstance(step, ToolStep):
        return f"tool={step.tool_name}"
    if isinstance(step, LoopStep):
        return f"max_iterations={step.max_iterations}"
    if isinstance(step, MemoryReadStep):
        return f"namespace={step.namespace}"
    if isinstance(step, MemoryWriteStep):
        return f"namespace={step.namespace}"
    if isinstance(step, DelegateBatchStep):
        return "batch delegate calls"
    if isinstance(step, DelegateStep | ModelStep | LogicStep):
        return None
    return None


def _route_label(route_key: object) -> str:
    """Return a normalized route label for a conditional edge."""
    if isinstance(route_key, str) and route_key.strip():
        return f"route={route_key.strip()}"
    return "route"


def _escape_label(label: str) -> str:
    """Escape Mermaid node and edge labels safely."""
    return html_escape(label, quote=True).replace("\n", "<br/>")
