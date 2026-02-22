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

from ._tools import ToolSpec

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
    """Message role used by chat-compatible backends."""
    content: str
    """Plain-text message content."""
    name: str | None = None
    """Optional participant name, when supported by the provider."""
    tool_call_id: str | None = None
    """Tool call identifier for tool-response messages."""
    tool_name: str | None = None
    """Tool name associated with a tool-response message."""


@dataclass(slots=True)
class LLMChatParams:
    """Provider-neutral generation controls passed with chat requests."""

    temperature: float | None = None
    """Sampling temperature override for the request."""
    max_tokens: int | None = None
    """Maximum number of output tokens to generate."""
    response_schema: dict[str, object] | None = None
    """Optional JSON schema describing required structured output."""
    provider_options: dict[str, object] = field(default_factory=dict)
    """Provider-specific raw options forwarded to backend adapters."""


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Tool-call intent emitted by a backend."""

    name: str
    """Resolved tool name selected by the model."""
    arguments_json: str
    """JSON-encoded argument payload to pass to the tool."""
    call_id: str
    """Stable call identifier used to pair call and result."""


@dataclass(slots=True, frozen=True)
class LLMToolResult:
    """Result payload used to feed tool outputs back into model turns."""

    call_id: str
    """Identifier of the originating tool call."""
    output_json: str
    """JSON-encoded tool output returned to the model."""
    error: str | None = None
    """Optional error text when tool execution failed."""


@dataclass(slots=True, frozen=True)
class Usage:
    """Token accounting information for an LLM call."""

    prompt_tokens: int | None = None
    """Prompt token count reported by the backend."""
    completion_tokens: int | None = None
    """Completion token count reported by the backend."""
    total_tokens: int | None = None
    """Total token count if reported by the backend."""


@dataclass(slots=True, frozen=True)
class TaskProfile:
    """Routing hints for selecting a backend."""

    priority: TaskPriority = "balanced"
    """Primary optimization objective for backend selection."""
    max_cost_usd: float | None = None
    """Upper cost bound for a single request, when enforced."""
    max_latency_ms: int | None = None
    """Upper latency target for a single request, when enforced."""
    tags: tuple[str, ...] = ()
    """Free-form tags used by model selection policies."""


@dataclass(slots=True, frozen=True)
class LLMRequest:
    """Provider-neutral request payload for LLM generation."""

    messages: Sequence[LLMMessage]
    """Ordered conversation/messages sent to the model."""
    model: str | None = None
    """Explicit model identifier override for this request."""
    temperature: float | None = None
    """Sampling temperature override."""
    max_tokens: int | None = None
    """Maximum output token limit."""
    tools: Sequence[ToolSpec] = ()
    """Tool specifications exposed for model tool-calling."""
    response_schema: dict[str, object] | None = None
    """Optional schema for structured output validation."""
    response_format: dict[str, object] | None = None
    """Provider-specific response-format hints."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Caller metadata forwarded for tracing and diagnostics."""
    provider_options: dict[str, object] = field(default_factory=dict)
    """Backend/provider-specific low-level options."""
    task_profile: TaskProfile | None = None
    """Optional routing profile used by selector-aware clients."""


@dataclass(slots=True, frozen=True)
class Provenance:
    """Provenance metadata for reproducibility and audit trails."""

    backend_name: str
    """Configured backend instance name."""
    backend_kind: str
    """Backend implementation family/type identifier."""
    model_id: str
    """Resolved model identifier used for the request."""
    base_url: str | None
    """Backend endpoint base URL, if network-backed."""
    started_at: str
    """ISO timestamp when request execution started."""
    completed_at: str
    """ISO timestamp when request execution completed."""
    config_hash: str
    """Stable hash of backend configuration inputs."""

    @staticmethod
    def now_iso() -> str:
        """Return the current UTC timestamp in ISO 8601 format.

        Returns:
            Current UTC timestamp as an ISO 8601 string.
        """
        return datetime.now(timezone.utc).isoformat()  # noqa: UP017


@dataclass(slots=True, frozen=True)
class LLMResponse:
    """Normalized non-streaming response payload returned by a backend."""

    text: str
    """Primary response text emitted by the model."""
    model: str | None = None
    """Model identifier reported by the backend."""
    provider: str | None = None
    """Provider/backend name that produced this response."""
    finish_reason: str | None = None
    """Provider-specific completion reason."""
    usage: Usage | dict[str, int] | None = None
    """Token usage counters when available."""
    latency_ms: int | None = None
    """End-to-end latency in milliseconds."""
    raw_output: dict[str, object] | None = None
    """Legacy/raw backend payload for debugging."""
    tool_calls: tuple[ToolCall, ...] = ()
    """Tool calls requested by the model in this response."""
    raw: dict[str, object] | None = None
    """Canonical raw backend payload snapshot."""
    provenance: Provenance | None = None
    """Execution provenance metadata for auditability."""


@dataclass(slots=True, frozen=True)
class LLMStreamEvent:
    """One event emitted from a streaming model response."""

    kind: LLMStreamEventKind
    """Event kind, either incremental delta or stream completion."""
    delta_text: str | None = None
    """Incremental text fragment for ``kind='delta'`` events."""
    response: LLMResponse | None = None
    """Final assembled response for ``kind='completed'`` events."""


@dataclass(slots=True, frozen=True)
class ToolCallDelta:
    """Incremental tool-call delta used for streaming responses."""

    call_id: str | None = None
    """Tool call id fragment or full id as streamed by provider."""
    name: str | None = None
    """Tool name fragment or full name for the streamed call."""
    arguments_json_delta: str | None = None
    """Incremental JSON argument text for the streamed call."""


@dataclass(slots=True, frozen=True)
class LLMDelta:
    """Incremental delta emitted by streaming model responses."""

    text_delta: str | None = None
    """Incremental text token/segment."""
    tool_call_delta: ToolCallDelta | None = None
    """Incremental tool-call payload, when provided by backend."""
    usage_delta: Usage | None = None
    """Incremental usage counters emitted mid-stream."""


@dataclass(slots=True, frozen=True)
class EmbeddingResult:
    """Embedding response payload returned by a backend."""

    vectors: Sequence[Sequence[float]]
    """Embedding vectors in request input order."""
    model_id: str | None = None
    """Model identifier used for embedding generation."""
    usage: Usage | None = None
    """Usage counters associated with the embedding request."""


@dataclass(slots=True, frozen=True)
class BackendCapabilities:
    """Capabilities supported by a backend."""

    streaming: bool
    """Whether backend supports incremental streaming responses."""
    tool_calling: ToolCallingMode
    """Tool-calling capability mode supported by backend."""
    json_mode: JSONMode
    """Structured JSON output mode supported by backend."""
    vision: bool
    """Whether backend accepts vision/image inputs."""
    max_context_tokens: int | None
    """Maximum context window tokens, if known."""


@dataclass(slots=True, frozen=True)
class BackendStatus:
    """Healthcheck status returned by a backend."""

    ok: bool
    """True when backend healthcheck succeeded."""
    message: str | None = None
    """Optional human-readable healthcheck summary."""
    details: Mapping[str, object] | None = None
    """Optional structured diagnostics from healthcheck."""
    checked_at: str | None = None
    """ISO timestamp for when the healthcheck was performed."""


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
        """Generate and return a full chat completion response.

        Args:
            messages: Ordered chat messages for this completion request.
            model: Target model identifier.
            params: Request controls such as temperature and max token limits.

        Returns:
            Normalized completion response payload.
        """

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Generate a streaming chat completion event sequence.

        Args:
            messages: Ordered chat messages for this completion request.
            model: Target model identifier.
            params: Request controls such as temperature and max token limits.

        Returns:
            Iterator of normalized streaming completion events.
        """

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate and return a full response from a request object.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Normalized completion response payload.
        """

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream a response from a request object.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Iterator of normalized response deltas.
        """

    def default_model(self) -> str:
        """Return default model identifier for the configured backend.

        Returns:
            Model identifier used when no model is supplied on a request.
        """


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
        """Generate one provider-backed chat response in normalized format.

        Args:
            messages: Ordered chat messages for this completion request.
            model: Target model identifier.
            params: Request controls such as temperature and max token limits.

        Returns:
            Normalized completion response payload.
        """

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream provider-backed chat events in normalized format.

        Args:
            messages: Ordered chat messages for this completion request.
            model: Target model identifier.
            params: Request controls such as temperature and max token limits.

        Returns:
            Iterator of normalized streaming completion events.
        """


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
