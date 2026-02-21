"""Shared builders for default structured-response schemas used by agents.

These helpers centralize JSON-schema payloads that agents pass to ``LLMChatParams``
for model-guided structured decisions. Keeping the schema construction in one
module avoids drift between agent implementations and makes runtime-fed schema
generation explicit and testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy


def clone_response_schema(schema: dict[str, object]) -> dict[str, object]:
    """Return a deep-cloned schema so per-call mutation cannot leak globally.

    Args:
        schema: JSON-schema-like mapping to clone.

    Returns:
        Deep copy of the schema mapping.
    """
    return deepcopy(schema)


def build_router_selection_response_schema(
    *,
    alternative_identifiers: Sequence[str],
) -> dict[str, object]:
    """Build schema for router route-selection output.

    The schema requires ``tool_names`` as a non-empty ordered list of route
    identifiers.

    Args:
        alternative_identifiers: Ordered route identifiers available to the router.

    Returns:
        JSON-schema-like mapping describing the router selection payload.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_names"],
        "properties": {
            "tool_names": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "enum": list(alternative_identifiers),
                },
            },
            "reason": {"type": "string"},
        },
    }


def build_tool_call_response_schema(
    *,
    tool_names: Sequence[str],
) -> dict[str, object]:
    """Build schema for one-step tool selection with structured arguments.

    Args:
        tool_names: Ordered tool identifiers available for selection.

    Returns:
        JSON-schema-like mapping describing the tool selection payload.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_name"],
        "properties": {
            "tool_name": {"type": "string", "enum": list(tool_names)},
            "tool_input": {"type": "object"},
            "reason": {"type": "string"},
        },
    }


def build_continuation_response_schema() -> dict[str, object]:
    """Build schema for multi-step continuation decisions.

    Returns:
        JSON-schema-like mapping describing the continuation payload.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["continue", "thought"],
        "properties": {
            "continue": {"type": "boolean"},
            "thought": {"type": "string"},
        },
    }


def build_multi_step_direct_controller_response_schema() -> dict[str, object]:
    """Build schema for direct-LLM multi-step controller decisions.

    Returns:
        JSON-schema-like mapping for structured continue/stop controller output.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "content"],
        "properties": {
            "decision": {"type": "string", "enum": ["CONTINUE", "STOP"]},
            "content": {"type": "string"},
            "final_output": {"type": "string"},
            "reason": {"type": "string"},
        },
    }


def build_multi_step_tool_router_response_schema(
    *,
    tool_names: Sequence[str],
) -> dict[str, object]:
    """Build schema for one multi-step tool-router decision.

    Args:
        tool_names: Allowed tool identifiers for controller-selected tool calls.

    Returns:
        JSON-schema-like mapping describing one router decision payload.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action"],
        "properties": {
            "action": {"type": "string", "enum": ["TOOL_CALL", "STOP"]},
            "tool_names": {
                "type": "array",
                "items": {"type": "string", "enum": list(tool_names)},
                "minItems": 1,
            },
            "tool_input": {"type": "object"},
            "final_output": {"type": "object"},
            "reason": {"type": "string"},
        },
    }
