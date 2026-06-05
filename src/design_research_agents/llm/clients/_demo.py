"""Workshop-friendly local demo client preset."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Mapping, Sequence

from design_research_agents._contracts._llm import (
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)

from ._shared import LlamaCppServerLLMClient

_DEMO_DEFAULT_PROVIDER_OPTIONS: dict[str, object] = {
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
}
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


class DemoLLMClient(LlamaCppServerLLMClient):
    """Workshop-friendly Qwen3 demo client backed by managed llama.cpp."""

    def __init__(
        self,
        *,
        name: str = "demo-local",
        model: str = "Qwen3-0.6B-Q8_0.gguf",
        hf_model_repo_id: str | None = "Qwen/Qwen3-0.6B-GGUF",
        api_model: str = "qwen3-0.6b-q8-demo",
        host: str = "127.0.0.1",
        port: int = 8001,
        context_window: int = 4096,
        startup_timeout_seconds: float = 120.0,
        request_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
        python_executable: str = sys.executable,
        extra_server_args: tuple[str, ...] = (),
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
        thinking: str = "off",
        default_temperature: float | None = 0.7,
        default_max_tokens: int | None = 256,
        default_provider_options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the managed local Qwen3 demo client.

        Args:
            name: Logical client name used in traces and provenance.
            model: GGUF filename or path for the managed llama.cpp server.
            hf_model_repo_id: Optional Hugging Face GGUF repository used for download/resolution.
            api_model: Model alias exposed by the OpenAI-compatible local server.
            host: Server bind host.
            port: Requested server port; a free fallback is selected when busy.
            context_window: llama.cpp context window size.
            startup_timeout_seconds: Maximum wait for the managed server to become ready.
            request_timeout_seconds: HTTP request timeout for model calls.
            poll_interval_seconds: Readiness-check polling interval during startup.
            python_executable: Python executable used to launch ``llama_cpp.server``.
            extra_server_args: Extra arguments forwarded to ``llama_cpp.server``.
            max_retries: Structured-output retry count for best-effort JSON/tool flows.
            model_patterns: Optional model id patterns accepted by this client.
            thinking: Qwen thinking control, one of ``"off"``, ``"on"``, or ``"auto"``.
            default_temperature: Temperature applied when a request omits one.
            default_max_tokens: Max-token cap applied when a request omits one.
            default_provider_options: Provider options merged before per-request overrides.

        Raises:
            ValueError: If ``thinking`` or default generation limits are invalid.
        """
        normalized_thinking = thinking.strip().lower()
        if normalized_thinking not in {"off", "on", "auto"}:
            raise ValueError("thinking must be one of: off, on, auto.")
        if default_max_tokens is not None and default_max_tokens <= 0:
            raise ValueError("default_max_tokens must be positive when supplied.")
        if default_temperature is not None and default_temperature < 0:
            raise ValueError("default_temperature must be non-negative when supplied.")

        provider_defaults = dict(_DEMO_DEFAULT_PROVIDER_OPTIONS)
        if default_provider_options is not None:
            provider_defaults.update(dict(default_provider_options))

        self._thinking = normalized_thinking
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._default_provider_options = provider_defaults

        super().__init__(
            name=name,
            model=model,
            hf_model_repo_id=hf_model_repo_id,
            api_model=api_model,
            host=host,
            port=port,
            context_window=context_window,
            startup_timeout_seconds=startup_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            python_executable=python_executable,
            extra_server_args=extra_server_args,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._config_snapshot.update(
            {
                "thinking": self._thinking,
                "default_temperature": self._default_temperature,
                "default_max_tokens": self._default_max_tokens,
                "default_provider_options": dict(self._default_provider_options),
            }
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one normalized demo response.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Response with Qwen think blocks removed from the visible text.
        """
        response = super().generate(self._normalize_request(request))
        return _strip_response_thinking(response)

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream one normalized demo response.

        Args:
            request: Provider-neutral request payload.

        Yields:
            A cleaned text delta for the generated response.
        """
        response = self.generate(request)
        if response.text:
            yield LLMDelta(text_delta=response.text)

    def _normalize_request(self, request: LLMRequest) -> LLMRequest:
        """Return a request with demo defaults and Qwen thinking controls applied.

        Args:
            request: Source request to normalize.

        Returns:
            Request copied with model, messages, sampling controls, and token cap defaults.
        """
        provider_options = dict(self._default_provider_options)
        provider_options.update(dict(request.provider_options))
        return LLMRequest(
            messages=_apply_thinking_control(request.messages, thinking=self._thinking),
            model=(request.model.strip() if request.model else self.default_model()),
            temperature=request.temperature if request.temperature is not None else self._default_temperature,
            max_tokens=request.max_tokens if request.max_tokens is not None else self._default_max_tokens,
            tools=request.tools,
            response_schema=request.response_schema,
            response_format=request.response_format,
            metadata=dict(request.metadata),
            provider_options=provider_options,
            task_profile=request.task_profile,
        )


def _apply_thinking_control(messages: Sequence[LLMMessage], *, thinking: str) -> tuple[LLMMessage, ...]:
    """Append Qwen thinking control tags to the last user message when requested.

    Args:
        messages: Source chat messages.
        thinking: Normalized thinking mode: ``"off"``, ``"on"``, or ``"auto"``.

    Returns:
        Messages with ``/no_think`` or ``/think`` appended when needed.
    """
    if thinking == "auto":
        return tuple(messages)
    control = "/no_think" if thinking == "off" else "/think"
    if any(control in message.content for message in messages):
        return tuple(messages)

    updated = list(messages)
    for index in range(len(updated) - 1, -1, -1):
        message = updated[index]
        if message.role != "user":
            continue
        updated[index] = LLMMessage(
            role=message.role,
            content=f"{message.content.rstrip()}\n\n{control}",
            name=message.name,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
        )
        return tuple(updated)

    updated.append(LLMMessage(role="user", content=control))
    return tuple(updated)


def _strip_response_thinking(response: LLMResponse) -> LLMResponse:
    """Return a response with Qwen think blocks removed from visible text.

    Args:
        response: Original backend response.

    Returns:
        Response copy whose ``text`` field hides closed ``<think>`` blocks.
    """
    cleaned_text = _strip_think_blocks(response.text)
    if cleaned_text == response.text:
        return response
    return LLMResponse(
        text=cleaned_text,
        model=response.model,
        provider=response.provider,
        finish_reason=response.finish_reason,
        usage=response.usage,
        latency_ms=response.latency_ms,
        raw_output=response.raw_output,
        tool_calls=response.tool_calls,
        raw=response.raw,
        provenance=response.provenance,
    )


def _strip_think_blocks(text: str) -> str:
    """Remove closed Qwen ``<think>`` blocks and stray think tags from text."""
    without_blocks = _THINK_BLOCK_PATTERN.sub("", text)
    without_tags = re.sub(r"</?think>", "", without_blocks, flags=re.IGNORECASE)
    return without_tags.strip()


__all__ = ["DemoLLMClient"]
