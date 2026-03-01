"""Deterministic HTML stand-in backend for offline demos and smoke checks."""

from __future__ import annotations

from collections.abc import Iterator
from html import escape
from typing import override

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm._backends._base import BaseLLMBackend

_FALLBACK_PROMPT = "Hello from design-research-agents."


class HTMLBackend(BaseLLMBackend):
    """Backend that wraps prompt text in a simple deterministic HTML response."""

    def __init__(self, *, name: str, model: str, config_hash: str) -> None:
        """Configure the HTML stand-in backend.

        Args:
            name: Stable backend name used in traces and diagnostics.
            model: Default model identifier exposed by the client.
            config_hash: Stable hash used for provenance metadata.
        """
        super().__init__(
            name=name,
            kind="html",
            default_model=model,
            base_url=None,
            config_hash=config_hash,
            max_retries=0,
            model_patterns=(model,),
        )

    @override
    def capabilities(self) -> BackendCapabilities:
        """Return capabilities for the HTML stand-in backend."""
        return BackendCapabilities(
            streaming=True,
            tool_calling="none",
            json_mode="none",
            vision=False,
            max_context_tokens=None,
        )

    @override
    def healthcheck(self) -> BackendStatus:
        """Return an always-healthy status for the HTML backend."""
        return BackendStatus(ok=True, message="HTML stand-in backend is always healthy.")

    @override
    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one deterministic HTML response."""
        text = _render_html(_resolve_prompt_text(request))
        return LLMResponse(text=text, model=request.model, provider=self.name)

    @override
    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream the full deterministic HTML payload as one delta."""
        response = self._generate(request)
        yield LLMDelta(text_delta=response.text)


def _resolve_prompt_text(request: LLMRequest) -> str:
    """Return the source text used to build the HTML response."""
    latest_user_content = ""
    for message in request.messages:
        content = message.content.strip()
        if message.role == "user" and content:
            latest_user_content = content
    if latest_user_content:
        return latest_user_content

    combined_content = " ".join(message.content.strip() for message in request.messages if message.content.strip())
    if combined_content:
        return combined_content
    return _FALLBACK_PROMPT


def _render_html(content: str) -> str:
    """Return the deterministic HTML wrapper for content."""
    escaped_content = escape(content, quote=True)
    return f"<html>\n  <body>\n    <h1>Mock Model Response</h1>\n    <p>{escaped_content}</p>\n  </body>\n</html>"
