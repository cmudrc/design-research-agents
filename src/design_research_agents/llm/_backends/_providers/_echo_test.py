"""Deterministic echo backend used for tests and smoke checks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import override

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm._backends._utils import messages_to_prompt


class EchoTestBackend(BaseLLMBackend):
    """Echo backend that returns a normalized prompt string."""

    def __init__(self, *, name: str, model: str, config_hash: str) -> None:
        """Configure a deterministic echo backend for tests.

        Args:
            name: Value supplied for ``name``.
            model: Value supplied for ``model``.
            config_hash: Value supplied for ``config_hash``.
        """
        super().__init__(
            name=name,
            kind="echo_test",
            default_model=model,
            base_url=None,
            config_hash=config_hash,
            max_retries=0,
            model_patterns=(model,),
        )

    @override
    def capabilities(self) -> BackendCapabilities:
        """Return capabilities exposed by the echo backend.

        Returns:
            Result produced by this call.
        """
        return BackendCapabilities(
            streaming=True,
            tool_calling="none",
            json_mode="none",
            vision=False,
            max_context_tokens=None,
        )

    @override
    def healthcheck(self) -> BackendStatus:
        """Return an always-healthy status for deterministic test backend.

        Returns:
            Result produced by this call.
        """
        return BackendStatus(ok=True, message="echo-test backend is always healthy.")

    @override
    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one deterministic echo response for the provided request.

        Args:
            request: Value supplied for ``request``.

        Returns:
            Result produced by this call.
        """
        prompt = messages_to_prompt(request.messages)
        cleaned_prompt = " ".join(prompt.strip().split())
        if not cleaned_prompt:
            cleaned_prompt = "Hello from design-research-agents."
        text = f"[{request.model}] {cleaned_prompt}"
        return LLMResponse(text=text, model=request.model, provider=self.name)

    @override
    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream one deterministic echo delta for the provided request.

        Args:
            request: Value supplied for ``request``.

        Yields:
            The yielded values.
        """
        response = self._generate(request)
        yield LLMDelta(text_delta=response.text)
