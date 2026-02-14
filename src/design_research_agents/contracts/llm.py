"""Provider-agnostic LLM interfaces, payloads, and normalized error taxonomy.

These contracts are shared by client implementations and backend adapters so
agent code can remain independent of any specific provider SDK.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

LLMRole = Literal["system", "user", "assistant", "tool"]
LLMStreamEventKind = Literal["delta", "completed"]


@dataclass(slots=True, frozen=True)
class LLMMessage:
    """One chat message in the provider-neutral completion format.

    Attributes:
        role: Semantic message role (system, user, assistant, or tool).
        content: Plain text content sent to the model.
        name: Optional participant name used by providers that support it.
    """

    role: LLMRole
    content: str
    name: str | None = None


@dataclass(slots=True)
class LLMChatParams:
    """Provider-neutral generation controls passed with chat requests.

    Attributes:
        temperature: Optional sampling temperature.
        max_tokens: Optional output token cap.
        response_schema: Optional JSON schema used for structured responses.
        provider_options: Backend-specific options carried through unchanged.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: dict[str, object] | None = None
    provider_options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Normalized non-streaming response payload returned by a backend adapter.

    Attributes:
        model: Model identifier used for the request.
        text: Final generated text content.
        provider: Optional backend/provider label.
        finish_reason: Optional provider stop reason.
        usage: Optional token accounting information.
        latency_ms: Optional wall-clock latency measurement.
        raw_output: Optional provider-native diagnostic payload.
    """

    model: str
    text: str
    provider: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    latency_ms: int | None = None
    raw_output: dict[str, object] | None = None


@dataclass(slots=True, frozen=True)
class LLMStreamEvent:
    """One event emitted from a streaming model response.

    Attributes:
        kind: Event type; either an incremental text delta or completion.
        delta_text: Incremental text for ``kind="delta"``.
        response: Final normalized response for ``kind="completed"``.
    """

    kind: LLMStreamEventKind
    delta_text: str | None = None
    response: LLMResponse | None = None


class LLMClient(Protocol):
    """Protocol implemented by provider-agnostic LLM clients.

    Callers can depend on this interface for both full-response and streaming
    generation paths without coupling to provider-specific SDK contracts.
    """

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate and return a full chat completion response.

        Implementations should normalize provider-specific payloads into the
        shared ``LLMResponse`` contract.
        """

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Generate a streaming chat completion event sequence.

        Streams should conclude with a ``kind="completed"`` event containing the
        final response payload.
        """


class LLMProviderAdapter(Protocol):
    """Backend adapter contract consumed by :class:`LLMClient` implementations.

    Adapters are responsible for translating provider-specific SDK behavior into
    normalized contract payloads and exceptions.
    """

    provider_name: str

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate one provider-backed chat response in normalized format.

        Adapters should translate provider-specific errors to contract exceptions.
        """

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream provider-backed chat events in normalized format.

        Event ordering and completion semantics should match ``LLMClient`` rules.
        """


class LLMError(Exception):
    """Base exception for provider-independent LLM runtime failures.

    Concrete subclasses classify common provider failure categories.
    """


class LLMAuthError(LLMError):
    """Authentication or authorization failure raised by provider backends.

    Typically indicates invalid credentials, missing keys, or permission denial.
    """


class LLMRateLimitError(LLMError):
    """Provider rate-limit failure indicating callers should throttle or retry.

    Callers should use retry/backoff policies appropriate for the provider.
    """


class LLMInvalidRequestError(LLMError):
    """Invalid request payload or unsupported provider/backend configuration.

    Raised when request shape, model selection, or backend setup is invalid.
    """


class LLMProviderError(LLMError):
    """General provider runtime failure not covered by specialized subclasses.

    Serves as catch-all for provider errors without stronger classification.
    """
