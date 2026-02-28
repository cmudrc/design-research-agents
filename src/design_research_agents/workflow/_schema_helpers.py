"""Convenience builders for workflow input/output schema declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

_SCALAR_TYPES = {"string", "number", "integer", "boolean", "null", "object", "array"}
type SchemaType = Literal["string", "number", "integer", "boolean", "null", "object", "array"]


def scalar(type_name: SchemaType) -> dict[str, object]:
    """Return a scalar-schema mapping for one JSON-schema-like type name."""
    normalized = str(type_name).strip()
    if normalized not in _SCALAR_TYPES:
        raise ValueError(f"Unsupported schema type '{type_name}'.")
    return {"type": normalized}


def list_of(item_schema: Mapping[str, object]) -> dict[str, object]:
    """Return an array schema whose ``items`` are constrained by ``item_schema``."""
    return {"type": "array", "items": dict(item_schema)}


def typed_dict(
    *,
    required: Mapping[str, Mapping[str, object]] | None = None,
    optional: Mapping[str, Mapping[str, object]] | None = None,
    additional_properties: bool = False,
) -> dict[str, object]:
    """Return an object schema from required/optional field schema mappings."""
    properties: dict[str, object] = {}
    required_fields: list[str] = []

    for field_name, field_schema in (required or {}).items():
        normalized_field_name = _normalize_field_name(field_name)
        properties[normalized_field_name] = _normalize_field_schema(field_schema)
        required_fields.append(normalized_field_name)
    for field_name, field_schema in (optional or {}).items():
        normalized_field_name = _normalize_field_name(field_name)
        if normalized_field_name in properties:
            raise ValueError(f"Duplicate schema field '{normalized_field_name}'.")
        properties[normalized_field_name] = _normalize_field_schema(field_schema)

    return {
        "type": "object",
        "properties": properties,
        "required": required_fields,
        "additionalProperties": bool(additional_properties),
    }


def _normalize_field_name(field_name: str) -> str:
    """Validate and normalize one object-schema field name."""
    normalized_field_name = str(field_name).strip()
    if not normalized_field_name:
        raise ValueError("Schema field names must be non-empty.")
    return normalized_field_name


def _normalize_field_schema(field_schema: Mapping[str, object]) -> dict[str, object]:
    """Normalize one field schema mapping."""
    return dict(field_schema)


__all__ = ["list_of", "scalar", "typed_dict"]
