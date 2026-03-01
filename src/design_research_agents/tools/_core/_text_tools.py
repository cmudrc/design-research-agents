"""Text utility tool family."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource

from ._helpers import get_str


def register_text_tools(source: InProcessToolSource) -> None:
    """Register core text analysis and extraction utilities.

    Args:
        source: Value supplied for ``source``.
    """
    metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_read=False, filesystem_write=False),
        timeout_s=5,
        max_output_bytes=65_536,
        risky=False,
    )

    source.register_tool(
        spec=ToolSpec(
            name="text.word_count",
            description="Count words, characters, lines, and unique words in text.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "char_count": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "line_count": {"type": "integer"},
                    "unique_word_count": {"type": "integer"},
                },
                "required": [
                    "char_count",
                    "word_count",
                    "line_count",
                    "unique_word_count",
                ],
            },
            metadata=metadata,
        ),
        handler=_word_count_handler,
    )

    source.register_tool(
        spec=ToolSpec(
            name="text.extract_json",
            description="Extract exactly one JSON object from a text blob.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"json": {"type": "object"}},
                "required": ["json"],
            },
            metadata=metadata,
        ),
        handler=_extract_json_tool_handler,
    )

    source.register_tool(
        spec=ToolSpec(
            name="text.diff",
            description="Compute a unified diff between two text strings.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "string"},
                },
                "required": ["a", "b"],
            },
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"],
            },
            metadata=metadata,
        ),
        handler=_diff_tool_handler,
    )


def _word_count_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Word count handler.

    Args:
        input_dict: Value supplied for ``input_dict``.
        request_id: Value supplied for ``request_id``.
        dependencies: Value supplied for ``dependencies``.

    Returns:
        Result produced by this call.
    """
    del request_id, dependencies
    text = get_str(input_dict, "text")
    words = [word for word in text.split() if word]
    normalized_words = {word.strip(".,!?;:").lower() for word in words if word.strip(".,!?;:")}
    line_count = 0 if not text else text.count("\n") + 1
    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": line_count,
        "unique_word_count": len(normalized_words),
    }


def _extract_json_tool_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Extract json tool handler.

    Args:
        input_dict: Value supplied for ``input_dict``.
        request_id: Value supplied for ``request_id``.
        dependencies: Value supplied for ``dependencies``.

    Returns:
        Result produced by this call.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    del request_id, dependencies
    text = get_str(input_dict, "text")
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return {"json": parsed}
    except json.JSONDecodeError:
        pass

    candidates = _extract_object_candidates(stripped)
    if len(candidates) != 1:
        raise ValueError("Unable to extract exactly one JSON object from text. Provide unambiguous JSON content.")
    parsed_candidate = json.loads(candidates[0])
    if not isinstance(parsed_candidate, dict):
        raise ValueError("Extracted JSON payload is not an object.")
    return {"json": parsed_candidate}


def _diff_tool_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Diff tool handler.

    Args:
        input_dict: Value supplied for ``input_dict``.
        request_id: Value supplied for ``request_id``.
        dependencies: Value supplied for ``dependencies``.

    Returns:
        Result produced by this call.
    """
    del request_id, dependencies
    a_text = get_str(input_dict, "a")
    b_text = get_str(input_dict, "b")
    diff_lines = difflib.unified_diff(
        a_text.splitlines(keepends=True),
        b_text.splitlines(keepends=True),
        fromfile="a",
        tofile="b",
    )
    return {"diff": "".join(diff_lines)}


def _extract_object_candidates(text: str) -> list[str]:
    """Extract object candidates.

    Args:
        text: Value supplied for ``text``.

    Returns:
        Result produced by this call.
    """
    candidates: list[str] = []
    depth = 0
    start_index: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index : index + 1])
                start_index = None
    return candidates
