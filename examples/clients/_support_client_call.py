"""Local helper for deterministic representative client call payloads."""

from __future__ import annotations

import os
from typing import Protocol

from design_research_agents.llm import LLMMessage, LLMRequest, LLMResponse


class _RequestLLMClient(Protocol):
    """Minimal protocol for request-based client examples."""

    def default_model(self) -> str: ...

    def generate(self, request: LLMRequest) -> LLMResponse: ...


def run_representative_chat(
    *,
    client: _RequestLLMClient,
    prompt: str,
    deterministic_response: str,
    system_prompt: str = "You are a concise engineering design assistant.",
    max_tokens: int = 120,
) -> dict[str, object]:
    """Run one representative request and return normalized call metadata."""
    deterministic_mode = os.environ.get("DRA_EXAMPLE_LLM_MODE", "").strip().lower() == "deterministic"
    model = client.default_model()
    if deterministic_mode:
        response_text = deterministic_response
        response_model = model
        response_provider = "deterministic"
    else:
        response: LLMResponse = client.generate(
            LLMRequest(
                messages=(
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=prompt),
                ),
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        )
        response_text = str(getattr(response, "text", ""))
        response_model = getattr(response, "model", None)
        response_provider = getattr(response, "provider", None)

    return {
        "execution_mode": "deterministic_stub" if deterministic_mode else "live_client",
        "prompt": prompt,
        "response_text": response_text,
        "response_model": response_model,
        "response_provider": response_provider,
        "response_has_text": bool(response_text.strip()),
    }
