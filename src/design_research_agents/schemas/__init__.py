"""JSON Schema loaders for public contract payload validation."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Final

from .validation import SchemaValidationError, validate_payload_against_schema

SCHEMA_NAMES: Final[tuple[str, ...]] = ("tool_spec", "tool_result", "execution_result")


def load_schema(name: str) -> dict[str, object]:
    """Load a JSON schema document from packaged resources.

    Args:
        name: Schema name without the ``.schema.json`` suffix.

    Returns:
        Parsed schema document as a mutable mapping.

    Raises:
        ValueError: If ``name`` is unsupported, or if the file does not deserialize into a JSON
            object.
    """
    if name not in SCHEMA_NAMES:
        raise ValueError(f"Unsupported schema '{name}'.")

    schema_resource = files("design_research_agents.schemas").joinpath(f"{name}.schema.json")
    with schema_resource.open("r", encoding="utf-8") as schema_file:
        loaded_schema = json.load(schema_file)
    if not isinstance(loaded_schema, dict):
        raise ValueError(f"Schema '{name}' must deserialize into an object.")

    return loaded_schema


__all__ = [
    "SCHEMA_NAMES",
    "SchemaValidationError",
    "load_schema",
    "validate_payload_against_schema",
]
