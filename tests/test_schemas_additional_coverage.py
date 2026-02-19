from __future__ import annotations

import pytest

from design_research_agents.schemas import load_schema
from design_research_agents.schemas import validation as validation_module
from design_research_agents.schemas.validation import (
    SchemaValidationError,
    validate_payload_against_schema,
)


def test_load_schema_rejects_unsupported_and_non_object_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Unsupported schema"):
        load_schema("unknown_schema")

    monkeypatch.setattr("design_research_agents.schemas.json.load", lambda _handle: ["bad"])
    with pytest.raises(ValueError, match="must deserialize into an object"):
        load_schema("tool_spec")


def test_validation_anyof_type_and_object_array_edge_paths() -> None:
    # Unknown schema types are ignored by design.
    validate_payload_against_schema(payload={"a": 1}, schema={"type": "unknown"})

    # anyOf ignores non-mapping entries and returns early on first success.
    validate_payload_against_schema(
        payload="ok",
        schema={
            "anyOf": [
                "not-a-schema",
                {"type": "string"},
            ]
        },
    )

    with pytest.raises(SchemaValidationError, match="did not satisfy anyOf constraints"):
        validate_payload_against_schema(
            payload=True,
            schema={
                "anyOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ]
            },
        )

    # Non-string property keys and non-mapping child schemas are skipped.
    validate_payload_against_schema(
        payload={"a": "value"},
        schema={
            "type": "object",
            "properties": {
                1: {"type": "string"},
                "a": "invalid-child-schema",
                "missing": {"type": "string"},
            },
        },
    )

    with pytest.raises(SchemaValidationError, match="expected array"):
        validate_payload_against_schema(payload="not-a-list", schema={"type": "array"})

    validate_payload_against_schema(payload=[1, 2, 3], schema={"type": "array", "items": "bad"})


def test_validation_internal_helpers_cover_non_mapping_anyof_candidates() -> None:
    validation_module._validate_any_of(
        payload={"k": "v"},
        schema={"anyOf": ["bad-candidate", {"type": "object"}]},
        location="$",
    )
