from __future__ import annotations

import pytest

from design_research_agents._schemas import load_schema
from design_research_agents._schemas._validation import (
    SchemaValidationError,
    validate_payload_against_schema,
)


def test_load_schema_rejects_unsupported_and_non_object_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Unsupported schema"):
        load_schema("unknown_schema")

    monkeypatch.setattr("design_research_agents._schemas.json.load", lambda _handle: ["bad"])
    with pytest.raises(ValueError, match="must deserialize into an object"):
        load_schema("tool_spec")


def test_validation_anyof_type_and_object_array_edge_paths() -> None:
    with pytest.raises(SchemaValidationError, match="invalid JSON schema"):
        validate_payload_against_schema(payload={"a": 1}, schema={"type": "unknown"})

    # Draft 2020-12 boolean schemas compose cleanly with object schemas.
    validate_payload_against_schema(
        payload="ok",
        schema={
            "anyOf": [
                False,
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

    validate_payload_against_schema(
        payload={"a": "value"},
        schema={
            "type": "object",
            "properties": {"a": {"type": "string"}},
        },
    )

    with pytest.raises(SchemaValidationError, match="expected array"):
        validate_payload_against_schema(payload="not-a-list", schema={"type": "array"})

    with pytest.raises(SchemaValidationError, match="invalid JSON schema"):
        validate_payload_against_schema(payload=[1, 2, 3], schema={"type": "array", "items": "bad"})
