"""Shared model-selection helpers used by agent ``run`` execution paths.

The helper centralizes precedence rules so every agent resolves models
consistently from run payload overrides, constructor defaults, client defaults,
and global backend configuration.
"""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.llm import resolve_default_model


def resolve_agent_model(
    *,
    llm_client: object,
    input_payload: Mapping[str, object],
    init_model: str | None,
) -> str:
    """Resolve a model name using deterministic precedence rules.

    Resolution order is:
    1. ``input_payload["model"]`` when present and non-empty,
    2. ``init_model`` from agent construction,
    3. ``llm_client.default_model()`` when available,
    4. package-level backend default model configuration.

    Args:
        llm_client: LLM client or compatible object that may define ``default_model``.
        input_payload: Normalized run input payload mapping.
        init_model: Optional model override from agent construction.

    Returns:
        Resolved model name to use for the run.
    """
    raw_input_model = input_payload.get("model")
    if isinstance(raw_input_model, str):
        normalized_input_model = raw_input_model.strip()
        if normalized_input_model:
            return normalized_input_model

    if isinstance(init_model, str):
        normalized_init_model = init_model.strip()
        if normalized_init_model:
            return normalized_init_model

    default_model_getter = getattr(llm_client, "default_model", None)
    if callable(default_model_getter):
        resolved_from_client = default_model_getter()
        if isinstance(resolved_from_client, str):
            normalized_client_model = resolved_from_client.strip()
            if normalized_client_model:
                return normalized_client_model

    return resolve_default_model()
