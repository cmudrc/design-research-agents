"""Provider-agnostic LLM interfaces, payloads, and normalized error taxonomy.

These contracts are shared across agent code and backend adapters so call sites
can stay provider-neutral while still supporting both chat-style and request-
object style execution paths.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol

from .tools import ToolSpec

LLMRole = Literal["system", "user", "assistant", "tool"]
LLMStreamEventKind = Literal["delta", "completed"]
ToolCallingMode = Literal["native", "best_effort", "none"]
JSONMode = Literal["native", "prompt+validate", "none"]
TaskPriority = Literal["latency", "quality", "cost", "balanced"]


@dataclass(slots=True, frozen=True)
class LLMMessage:
    """One chat message in the provider-neutral completion format.

    Attributes:
        role: Semantic message role (system, user, assistant, or tool).
        content: Plain text content sent to the model.
        name: Optional participant name used by providers that support it.
        tool_call_id: Optional tool call id for tool-result messages.
        tool_name: Optional tool name for tool-result messages.
    """

    role: LLMRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(slots=True)
class LLMChatParams:
    """Provider-neutral generation controls passed with chat requests."""

    temperature: float | None = None
    max_tokens: int | None = None
    response_schema: dict[str, object] | None = None
    provider_options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Tool-call intent emitted by a backend."""

    name: str
    arguments_json: str
    call_id: str


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Result payload used to feed tool outputs back into model turns."""

    call_id: str
    output_json: str
    error: str | None = None


@dataclass(slots=True, frozen=True)
class Usage:
    """Token accounting information for an LLM call."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class TaskProfile:
    """Routing hints for selecting a backend."""

    priority: TaskPriority = "balanced"
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None
    tags: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class LLMRequest:
    """Provider-neutral request payload for LLM generation."""

    messages: Sequence[LLMMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: Sequence[ToolSpec] = ()
    response_schema: dict[str, object] | None = None
    response_format: dict[str, object] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    provider_options: dict[str, object] = field(default_factory=dict)
    task_profile: TaskProfile | None = None


@dataclass(slots=True, frozen=True)
class Provenance:
    """Provenance metadata for reproducibility and audit trails."""

    backend_name: str
    backend_kind: str
    model_id: str
    base_url: str | None
    started_at: str
    completed_at: str
    config_hash: str

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()  # noqa: UP017


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Normalized non-streaming response payload returned by a backend."""

    text: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    usage: Usage | dict[str, int] | None = None
    latency_ms: int | None = None
    raw_output: dict[str, object] | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    raw: dict[str, object] | None = None
    provenance: Provenance | None = None


@dataclass(slots=True, frozen=True)
class LLMStreamEvent:
    """One event emitted from a streaming model response."""

    kind: LLMStreamEventKind
    delta_text: str | None = None
    response: LLMResponse | None = None


@dataclass(slots=True, frozen=True)
class ToolCallDelta:
    """Incremental tool-call delta used for streaming responses."""

    call_id: str | None = None
    name: str | None = None
    arguments_json_delta: str | None = None


@dataclass(slots=True, frozen=True)
class LLMDelta:
    """Incremental delta emitted by streaming model responses."""

    text_delta: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    usage_delta: Usage | None = None


@dataclass(slots=True, frozen=True)
class EmbeddingResult:
    """Embedding response payload returned by a backend."""

    vectors: Sequence[Sequence[float]]
    model_id: str | None = None
    usage: Usage | None = None


@dataclass(slots=True, frozen=True)
class BackendCapabilities:
    """Capabilities supported by a backend."""

    streaming: bool
    tool_calling: ToolCallingMode
    json_mode: JSONMode
    vision: bool
    max_context_tokens: int | None


@dataclass(slots=True, frozen=True)
class BackendStatus:
    """Healthcheck status returned by a backend."""

    ok: bool
    message: str | None = None
    details: Mapping[str, object] | None = None
    checked_at: str | None = None


class LLMClient(Protocol):
    """Protocol implemented by provider-agnostic LLM clients.

    Implementations may support one or both call styles used in this package:
    chat-style methods (``chat``/``stream_chat``) and request-object methods
    (``generate``/``stream``).
    """

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate and return a full chat completion response."""

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Generate a streaming chat completion event sequence."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate and return a full response from a request object."""

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream a response from a request object."""

    def default_model(self) -> str:
        """Return default model identifier for the configured backend."""


class LLMProviderAdapter(Protocol):
    """Backend adapter contract consumed by ``LLMClient`` implementations."""

    provider_name: str

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate one provider-backed chat response in normalized format."""

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream provider-backed chat events in normalized format."""


class LLMError(Exception):
    """Base exception for provider-independent LLM runtime failures."""


class LLMAuthError(LLMError):
    """Authentication or authorization failure raised by provider backends."""


class LLMRateLimitError(LLMError):
    """Provider rate-limit failure indicating callers should throttle or retry."""


class LLMInvalidRequestError(LLMError):
    """Invalid request payload or unsupported provider/backend configuration."""


class LLMProviderError(LLMError):
    """General provider runtime failure not covered by specialized subclasses."""


class LLMBadResponseError(LLMError):
    """Raised when a provider returns an invalid or empty response payload."""


class LLMCapabilityError(LLMError):
    """Raised when a backend cannot satisfy required capabilities."""
