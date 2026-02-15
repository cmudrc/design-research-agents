"""Shared model-resolution helpers used by agent execution paths."""

from __future__ import annotations


def resolve_agent_model(
    *,
    llm_client: object,
) -> str:
    """Resolve model name strictly from ``llm_client.default_model()``.

    Args:
        llm_client: LLM client that must expose ``default_model()``.

    Returns:
        Non-empty model identifier.

    Raises:
        ValueError: If ``default_model()`` is missing or returns an empty value.
    """
    default_model_getter = getattr(llm_client, "default_model", None)
    if not callable(default_model_getter):
        raise ValueError("LLM client must expose default_model().")
    resolved_model = default_model_getter()
    if not isinstance(resolved_model, str):
        raise ValueError("LLM client default_model() must return a string.")
    normalized_model = resolved_model.strip()
    if not normalized_model:
        raise ValueError("LLM client default_model() returned an empty model id.")
    return normalized_model
