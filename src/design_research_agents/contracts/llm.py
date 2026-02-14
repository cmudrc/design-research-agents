"""Provider-agnostic LLM interfaces and payload contracts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

LLMRole = Literal["system", "user", "assistant", "tool"]
LLMStreamEventKind = Literal["delta", "completed"]


@dataclass(slots=True, frozen=True)
class LLMMessage:
    """A single chat message for model completion."""

    role: LLMRole
    content: str
    name: str | None = None


@dataclass(slots=True)
class LLMChatParams:
    """Common generation controls shared across providers."""

    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: dict[str, object] | None = None
    provider_options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Final non-streaming model response."""

    model: str
    text: str
    provider: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    latency_ms: int | None = None
    raw_output: dict[str, object] | None = None


@dataclass(slots=True, frozen=True)
class LLMStreamEvent:
    """Single event emitted by a streaming model response."""

    kind: LLMStreamEventKind
    delta_text: str | None = None
    response: LLMResponse | None = None


class LLMClient(Protocol):
    """Protocol that all LLM providers should satisfy."""

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate a full chat completion response."""

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Generate a streaming chat completion response."""


class LLMProviderAdapter(Protocol):
    """Backend adapter contract used by :class:`LLMClient` implementations."""

    provider_name: str

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate one provider-native chat response."""

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream provider-native chat events."""


class LLMError(Exception):
    """Base class for provider-independent LLM runtime failures."""


class LLMAuthError(LLMError):
    """Authentication or authorization failure."""


class LLMRateLimitError(LLMError):
    """Provider rate limit failure."""


class LLMInvalidRequestError(LLMError):
    """Invalid request payload or unsupported provider configuration."""


class LLMProviderError(LLMError):
    """General provider runtime failure."""
