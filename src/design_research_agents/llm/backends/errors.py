"""Shared error mapping helpers for backend implementations."""

from __future__ import annotations

from design_research_agents.contracts.llm import (
    LLMAuthError,
    LLMError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
)


def map_backend_exception(exc: Exception) -> LLMError:
    """Map backend/provider exceptions into normalized contract exceptions.

    Args:
        exc: Parameter value.

    Returns:
        The resulting value.
    """
    if isinstance(exc, LLMError):
        return exc

    message = str(exc)
    lower_message = message.lower()
    status_code = getattr(exc, "status_code", None)

    if status_code in {401, 403}:
        return LLMAuthError(message)
    if status_code == 429:
        return LLMRateLimitError(message)
    if status_code in {400, 404, 422}:
        return LLMInvalidRequestError(message)

    if "api key" in lower_message or "not set" in lower_message:
        return LLMAuthError(message)
    if "rate limit" in lower_message:
        return LLMRateLimitError(message)
    if isinstance(exc, ValueError):
        return LLMInvalidRequestError(message)

    return LLMProviderError(message)
