"""Centralized termination-reason and decision-source constants."""

from __future__ import annotations

SOURCE_MODEL = "model"
"""Decision source for model-produced structured outputs."""

SOURCE_GUARDRAIL = "guardrail"
"""Decision source for guardrail-enforced outcomes."""

SOURCE_INVALID_PAYLOAD = "invalid_payload"
"""Decision source for invalid/unparseable structured payloads."""

TERMINATED_APPROVED = "approved"
"""Termination reason when critique loop approves the proposal."""

TERMINATED_COMPLETED = "completed"
"""Termination reason when the execution plan completes successfully."""

TERMINATED_CONTROLLER_INVALID_PAYLOAD = "controller_invalid_payload"
"""Termination reason when controller payload is invalid."""

TERMINATED_CONTINUATION_INVALID_PAYLOAD = "continuation_invalid_payload"
"""Termination reason when continuation payload is invalid."""

TERMINATED_INVALID_ROUTE_SELECTION = "invalid_route_selection"
"""Termination reason when selected route does not resolve to a known alternative."""

TERMINATED_INVALID_STEP_OUTPUT = "invalid_step_output"
"""Termination reason when one step output payload is invalid."""

TERMINATED_MAX_ITERATIONS_REACHED = "max_iterations_reached"
"""Termination reason when the iteration cap is reached."""

TERMINATED_MAX_STEPS_REACHED = "max_steps_reached"
"""Termination reason when step cap is reached."""

TERMINATED_ROUTING_FAILURE = "routing_failure"
"""Termination reason when routing fails before delegate execution."""

TERMINATED_STEP_FAILURE = "step_failure"
"""Termination reason when one execution step fails."""

TERMINATED_UNKNOWN_ALTERNATIVE = "unknown_alternative"
"""Termination reason when selected alternative is not registered."""


def continuation_stopped_reason(source: str) -> str:
    """Build continuation stop reason string from source label.

    Args:
        source: Continuation decision source.

    Returns:
        Prefixed continuation stop reason.
    """
    return f"continuation_stopped:{source}"


def stop_reason(source: str) -> str:
    """Build stop reason string from source label.

    Args:
        source: Stop decision source.

    Returns:
        Prefixed stop reason.
    """
    return f"stop:{source}"


__all__ = [
    "SOURCE_GUARDRAIL",
    "SOURCE_INVALID_PAYLOAD",
    "SOURCE_MODEL",
    "TERMINATED_APPROVED",
    "TERMINATED_COMPLETED",
    "TERMINATED_CONTINUATION_INVALID_PAYLOAD",
    "TERMINATED_CONTROLLER_INVALID_PAYLOAD",
    "TERMINATED_INVALID_ROUTE_SELECTION",
    "TERMINATED_INVALID_STEP_OUTPUT",
    "TERMINATED_MAX_ITERATIONS_REACHED",
    "TERMINATED_MAX_STEPS_REACHED",
    "TERMINATED_ROUTING_FAILURE",
    "TERMINATED_STEP_FAILURE",
    "TERMINATED_UNKNOWN_ALTERNATIVE",
    "continuation_stopped_reason",
    "stop_reason",
]
