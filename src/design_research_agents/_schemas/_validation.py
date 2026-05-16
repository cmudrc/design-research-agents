"""Validation helpers backed by JSON Schema Draft 2020-12."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from jsonschema import Draft202012Validator, SchemaError


class SchemaValidationError(ValueError):
    """Raised when a payload fails JSON Schema validation."""


def validate_payload_against_schema(
    *,
    payload: object,
    schema: Mapping[str, object] | None,
    location: str = "$",
) -> None:
    """Validate a payload against a JSON Schema document.

    Args:
        payload: Arbitrary payload value to validate.
        schema: Schema mapping to validate against.
        location: Location label used for error messages.

    Raises:
        SchemaValidationError: If validation fails.
    """
    if schema is None:
        return
    if not isinstance(schema, Mapping):
        raise SchemaValidationError(f"{location}: schema must be a mapping")
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.path))
    except SchemaError as exc:
        raise SchemaValidationError(f"{location}: invalid JSON schema: {exc.message}") from exc
    if not errors:
        return
    raise SchemaValidationError(_format_validation_error(errors[0], location=location))


def _format_validation_error(error: object, *, location: str) -> str:
    path = getattr(error, "path", ())
    validator = getattr(error, "validator", "")
    message = str(getattr(error, "message", error))
    resolved_location = _join_location(location, path)

    if validator == "additionalProperties":
        return f"{resolved_location}: unexpected field ({message})"
    if validator == "anyOf":
        return f"{resolved_location}: payload did not satisfy anyOf constraints ({message})"
    if validator == "enum":
        instance = getattr(error, "instance", None)
        enum_values = getattr(error, "validator_value", ())
        return f"{resolved_location}: value {instance!r} is not in enum {list(enum_values)!r}"
    if validator == "required":
        return f"{resolved_location}: required field missing ({message})"
    if validator == "type":
        expected = getattr(error, "validator_value", None)
        if isinstance(expected, str):
            return f"{resolved_location}: expected {expected}"
        return f"{resolved_location}: type mismatch ({message})"
    return f"{resolved_location}: {message}"


def _join_location(base: str, path: Iterable[object]) -> str:
    location = base
    for part in path:
        if isinstance(part, int):
            location += f"[{part}]"
        else:
            location += f".{part}"
    return location
