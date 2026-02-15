"""Versioned JSON Schema loaders for public contract payload validation.

Schemas are shipped as package resources under explicit version directories so
callers can load stable schema documents for serialization and validation.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Final

SCHEMA_VERSION: Final[str] = "v1"
SCHEMA_NAMES: Final[tuple[str, ...]] = ("tool_spec", "tool_result", "agent_result")


def load_schema(name: str, *, version: str = SCHEMA_VERSION) -> dict[str, object]:
    """Load a JSON schema document from packaged resources.

    Args:
        name: Schema name without the ``.schema.json`` suffix.
        version: Version directory under ``design_research_agents.schemas``.

    Returns:
        Parsed schema document as a mutable mapping.

    Raises:
        ValueError: If ``version`` or ``name`` are unsupported, or if the file
            does not deserialize into a JSON object.
    """
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version '{version}'.")
    if name not in SCHEMA_NAMES:
        raise ValueError(f"Unsupported schema '{name}'.")

    schema_resource = files("design_research_agents.schemas").joinpath(
        version, f"{name}.schema.json"
    )
    with schema_resource.open("r", encoding="utf-8") as schema_file:
        loaded_schema = json.load(schema_file)
    if not isinstance(loaded_schema, dict):
        raise ValueError(f"Schema '{name}' must deserialize into an object.")

    return loaded_schema


__all__ = [
    "SCHEMA_NAMES",
    "SCHEMA_VERSION",
]
