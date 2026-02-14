"""Shared backend type definitions."""

from __future__ import annotations

from typing import Final, Literal, cast

# Literal keeps backend names precise at type-check time.
BackendName = Literal["echo-test", "openai", "llama-cpp-server"]
# Runtime tuple is reused for validation and CLI choices.
SUPPORTED_BACKENDS: Final[tuple[BackendName, ...]] = (
    "echo-test",
    "openai",
    "llama-cpp-server",
)


def parse_backend(value: str) -> BackendName:
    """Parse a backend value into a normalized backend string.

    Args:
        value: Backend name string.

    Returns:
        Normalized backend name.

    Raises:
        ValueError: If ``value`` does not map to a supported backend.
    """
    # Normalize CLI/user input before checking the literal allow-list.
    normalized = value.strip().lower()
    if normalized in SUPPORTED_BACKENDS:
        return cast(BackendName, normalized)

    valid = ", ".join(SUPPORTED_BACKENDS)
    raise ValueError(f"Unsupported backend '{value}'. Supported values are: {valid}.")
