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

    The schema allows either:
    - an integer selection index within runtime route bounds, or
    - a string selection identifier matching one runtime route tool name.

    Args:
        alternative_identifiers: Ordered route identifiers available to the router.

    Returns:
        JSON-schema-like mapping describing the router selection payload.
    """
    max_index = max(0, len(alternative_identifiers) - 1)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selection"],
        "properties": {
            "selection": {
                "anyOf": [
                    {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": max_index,
                    },
                    {
                        "type": "string",
                        "enum": list(alternative_identifiers),
                    },
                ]
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
        "required": ["tool_name", "tool_input"],
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
        "required": ["continue"],
        "properties": {
            "continue": {"type": "boolean"},
            "thought": {"type": "string"},
            "reason": {"type": "string"},
        },
    }
