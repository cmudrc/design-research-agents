"""Validation helpers for a constrained JSON-schema-like subset.

The validator intentionally supports only the subset used by this package:
- ``type`` (object, array, string, number, integer, boolean, null)
- ``required``
- ``properties``
- ``additionalProperties``
- ``items``
- ``enum``
- ``anyOf``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class SchemaValidationError(ValueError):
    """Raised when a payload fails schema-subset validation."""


def validate_payload_against_schema(
    *,
    payload: object,
    schema: Mapping[str, object] | None,
    location: str = "$",
) -> None:
    """Validate a payload against the constrained schema subset.

    Args:
        payload: Arbitrary payload value to validate.
        schema: Schema mapping to validate against.
        location: Location label used for error messages.

    Raises:
        SchemaValidationError: If validation fails.
    """
    if schema is None:
        return
    _validate(payload=payload, schema=schema, location=location)


def _validate(*, payload: object, schema: Mapping[str, object], location: str) -> None:
    any_of = schema.get("anyOf")
    if isinstance(any_of, Sequence) and not isinstance(any_of, (str, bytes)):
        errors: list[str] = []
        for candidate in any_of:
            if not isinstance(candidate, Mapping):
                continue
            try:
                _validate(payload=payload, schema=candidate, location=location)
            except SchemaValidationError as exc:
                errors.append(str(exc))
                continue
            return
        if errors:
            raise SchemaValidationError(
                f"{location}: payload did not satisfy anyOf constraints ({'; '.join(errors[:3])})"
            )

    enum_values = schema.get("enum")
    if (
        isinstance(enum_values, Sequence)
        and not isinstance(enum_values, (str, bytes))
        and payload not in enum_values
    ):
        raise SchemaValidationError(
            f"{location}: value {payload!r} is not in enum {list(enum_values)!r}"
        )

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        _validate_type(payload=payload, expected_type=expected_type, location=location)
    elif isinstance(expected_type, Sequence) and not isinstance(expected_type, (str, bytes)):
        type_errors: list[str] = []
        for candidate_type in expected_type:
            if not isinstance(candidate_type, str):
                continue
            try:
                _validate_type(payload=payload, expected_type=candidate_type, location=location)
            except SchemaValidationError as exc:
                type_errors.append(str(exc))
                continue
            break
        else:
            joined = "; ".join(type_errors[:3])
            raise SchemaValidationError(f"{location}: type mismatch ({joined})")

    if isinstance(expected_type, str) and expected_type == "object":
        _validate_object(payload=payload, schema=schema, location=location)
    if isinstance(expected_type, str) and expected_type == "array":
        _validate_array(payload=payload, schema=schema, location=location)


def _validate_type(*, payload: object, expected_type: str, location: str) -> None:
    if expected_type == "object":
        if not isinstance(payload, Mapping):
            raise SchemaValidationError(f"{location}: expected object")
        return
    if expected_type == "array":
        if not isinstance(payload, list):
            raise SchemaValidationError(f"{location}: expected array")
        return
    if expected_type == "string":
        if not isinstance(payload, str):
            raise SchemaValidationError(f"{location}: expected string")
        return
    if expected_type == "number":
        if isinstance(payload, bool) or not isinstance(payload, (int, float)):
            raise SchemaValidationError(f"{location}: expected number")
        return
    if expected_type == "integer":
        if isinstance(payload, bool) or not isinstance(payload, int):
            raise SchemaValidationError(f"{location}: expected integer")
        return
    if expected_type == "boolean":
        if not isinstance(payload, bool):
            raise SchemaValidationError(f"{location}: expected boolean")
        return
    if expected_type == "null" and payload is not None:
        raise SchemaValidationError(f"{location}: expected null")


def _validate_object(*, payload: object, schema: Mapping[str, object], location: str) -> None:
    if not isinstance(payload, Mapping):
        raise SchemaValidationError(f"{location}: expected object")

    required = schema.get("required")
    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        for field in required:
            if isinstance(field, str) and field not in payload:
                raise SchemaValidationError(f"{location}.{field}: required field missing")

    properties = schema.get("properties")
    typed_properties = properties if isinstance(properties, Mapping) else {}
    if schema.get("additionalProperties") is False:
        for key in payload:
            if key not in typed_properties:
                raise SchemaValidationError(f"{location}.{key}: unexpected field")

    for key, child_schema in typed_properties.items():
        if not isinstance(key, str):
            continue
        if key not in payload:
            continue
        if not isinstance(child_schema, Mapping):
            continue
        _validate(
            payload=payload[key],
            schema=child_schema,
            location=f"{location}.{key}",
        )


def _validate_array(*, payload: object, schema: Mapping[str, object], location: str) -> None:
    if not isinstance(payload, list):
        raise SchemaValidationError(f"{location}: expected array")
    items_schema = schema.get("items")
    if not isinstance(items_schema, Mapping):
        return
    for index, item in enumerate(payload):
        _validate(
            payload=item,
            schema=items_schema,
            location=f"{location}[{index}]",
        )
