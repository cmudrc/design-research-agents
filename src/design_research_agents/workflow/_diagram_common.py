"""Shared workflow-diagram helpers used by Mermaid and SVG renderers."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape as html_escape
from typing import cast

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
    if isinstance(step, DelegateStep):
        return f"delegate={type(step.delegate).__name__}"
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


def _delegate_nested_steps(step: WorkflowStep) -> tuple[WorkflowStep, ...] | None:
    """Return nested workflow steps for a delegate step when available."""
    if not isinstance(step, DelegateStep):
        return None

    nested_steps = _extract_workflow_steps(step.delegate)
    if nested_steps is not None:
        return nested_steps

    static_prompt = step.prompt.strip() if isinstance(step.prompt, str) else ""
    if not static_prompt:
        return None

    compile_callable = getattr(step.delegate, "compile", None)
    if not callable(compile_callable):
        return None

    try:
        compiled_execution = compile_callable(
            static_prompt,
            request_id="diagram:delegate",
            dependencies={},
        )
    except Exception:
        return _extract_workflow_steps(step.delegate)

    nested_steps = _extract_workflow_steps(getattr(compiled_execution, "workflow", None))
    if nested_steps is not None:
        return nested_steps
    return _extract_workflow_steps(step.delegate)


def _extract_workflow_steps(value: object) -> tuple[WorkflowStep, ...] | None:
    """Return workflow steps from a workflow-like object when available."""
    maybe_steps = getattr(value, "_steps", None)
    if _is_workflow_step_sequence(maybe_steps):
        return tuple(cast(Sequence[WorkflowStep], maybe_steps))

    maybe_workflow = getattr(value, "workflow", None)
    maybe_nested_steps = getattr(maybe_workflow, "_steps", None)
    if _is_workflow_step_sequence(maybe_nested_steps):
        return tuple(cast(Sequence[WorkflowStep], maybe_nested_steps))
    return None


def _is_workflow_step_sequence(value: object) -> bool:
    """Return whether a value looks like a stored workflow-step sequence."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return False
    return all(
        isinstance(
            item,
            ToolStep
            | DelegateStep
            | ModelStep
            | DelegateBatchStep
            | LogicStep
            | LoopStep
            | MemoryReadStep
            | MemoryWriteStep,
        )
        for item in value
    )
