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

from collections.abc import Callable, Mapping, Sequence
from typing import TypeGuard


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
    """Validate one payload node against one schema node.

    Args:
        payload: Payload value at the current location.
        schema: Schema mapping that constrains the payload.
        location: JSONPath-like location string used in error messages.
    """
    _validate_any_of(payload=payload, schema=schema, location=location)
    _validate_enum(payload=payload, schema=schema, location=location)

    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        _validate_type(payload=payload, expected_type=expected_type, location=location)
    elif _is_sequence(expected_type):
        _validate_union_types(payload=payload, expected_types=expected_type, location=location)

    if isinstance(expected_type, str) and expected_type == "object":
        _validate_object(payload=payload, schema=schema, location=location)
    if isinstance(expected_type, str) and expected_type == "array":
        _validate_array(payload=payload, schema=schema, location=location)


def _validate_type(*, payload: object, expected_type: str, location: str) -> None:
    """Validate that a payload matches one primitive/object/array schema type.

    Args:
        payload: Payload value to type-check.
        expected_type: Schema ``type`` value to enforce.
        location: JSONPath-like location string used in error messages.

    Raises:
        SchemaValidationError: If payload does not match ``expected_type``.
    """
    validators: dict[str, Callable[[object], bool]] = {
        "object": lambda value: isinstance(value, Mapping),
        "array": lambda value: isinstance(value, list),
        "string": lambda value: isinstance(value, str),
        "number": lambda value: not isinstance(value, bool) and isinstance(value, (int, float)),
        "integer": lambda value: not isinstance(value, bool) and isinstance(value, int),
        "boolean": lambda value: isinstance(value, bool),
        "null": lambda value: value is None,
    }
    validator = validators.get(expected_type)
    if validator is None:
        return
    if validator(payload):
        return
    raise SchemaValidationError(f"{location}: expected {expected_type}")


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Check whether a value is a non-string sequence.

    Args:
        value: Candidate value to inspect.

    Returns:
        ``True`` when ``value`` is a sequence and not ``str``/``bytes``.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _validate_any_of(*, payload: object, schema: Mapping[str, object], location: str) -> None:
    """Validate ``anyOf`` constraints by accepting the first matching branch.

    Args:
        payload: Payload value to validate.
        schema: Schema mapping that may contain ``anyOf``.
        location: JSONPath-like location string used in error messages.

    Raises:
        SchemaValidationError: If no ``anyOf`` branch validates the payload.
    """
    any_of = schema.get("anyOf")
    if not _is_sequence(any_of):
        return
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
        raise SchemaValidationError(f"{location}: payload did not satisfy anyOf constraints ({'; '.join(errors[:3])})")


def _validate_enum(*, payload: object, schema: Mapping[str, object], location: str) -> None:
    """Validate enum membership when ``enum`` is present in schema.

    Args:
        payload: Payload value to validate.
        schema: Schema mapping that may contain ``enum`` values.
        location: JSONPath-like location string used in error messages.

    Raises:
        SchemaValidationError: If payload is not in the allowed enum set.
    """
    enum_values = schema.get("enum")
    if not _is_sequence(enum_values):
        return
    if payload in enum_values:
        return
    raise SchemaValidationError(f"{location}: value {payload!r} is not in enum {list(enum_values)!r}")


def _validate_union_types(
    *,
    payload: object,
    expected_types: Sequence[object],
    location: str,
) -> None:
    """Validate payload against a union of candidate schema types.

    Args:
        payload: Payload value to validate.
        expected_types: Candidate schema ``type`` entries.
        location: JSONPath-like location string used in error messages.

    Raises:
        SchemaValidationError: If payload matches none of the candidate types.
    """
    type_errors: list[str] = []
    for candidate_type in expected_types:
        if not isinstance(candidate_type, str):
            continue
        try:
            _validate_type(payload=payload, expected_type=candidate_type, location=location)
        except SchemaValidationError as exc:
            type_errors.append(str(exc))
            continue
        return
    joined = "; ".join(type_errors[:3])
    raise SchemaValidationError(f"{location}: type mismatch ({joined})")


def _validate_object(*, payload: object, schema: Mapping[str, object], location: str) -> None:
    """Validate object payloads, required keys, and child properties.

    Args:
        payload: Payload value expected to be an object mapping.
        schema: Object schema containing ``properties`` and related rules.
        location: JSONPath-like location string used in error messages.

    Raises:
        SchemaValidationError: If object constraints are violated.
    """
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
    """Validate array payloads and recursively validate each item.

    Args:
        payload: Payload value expected to be a list.
        schema: Array schema that may include an ``items`` schema.
        location: JSONPath-like location string used in error messages.

    Raises:
        SchemaValidationError: If payload is not an array or item validation fails.
    """
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
