"""SGLang local backend using OpenAI-compatible HTTP endpoints."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Sequence
from http.client import HTTPResponse
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    BackendStatus,
    LLMAuthError,
    LLMDelta,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    ToolCallDelta,
)
from design_research_agents.llm.backends.base import BaseLLMBackend
from design_research_agents.llm.backends.utils import (
    parse_tool_calls,
    parse_usage,
)

from .sglang_server import SglangServerBackend

_SGLANG_CAPABILITIES = BackendCapabilities(
    streaming=True,
    tool_calling="best_effort",
    json_mode="prompt+validate",
    vision=False,
    max_context_tokens=None,
)


class SglangLocalBackend(BaseLLMBackend):
    """Backend for local/self-hosted SGLang OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        default_model: str,
        request_timeout_seconds: float,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
        managed_server: SglangServerBackend | None = None,
    ) -> None:
        """Initialize SGLang backend transport and optional managed server.

        Args:
            name: Unique backend name for tracing and routing.
            base_url: Base URL for the OpenAI-compatible SGLang endpoint.
            default_model: Default model id used when request model is omitted.
            request_timeout_seconds: HTTP request timeout in seconds.
            config_hash: Stable hash of backend config inputs.
            max_retries: Maximum retry attempts for generation requests.
            model_patterns: Optional glob-like model patterns for selector routing.
            managed_server: Optional managed SGLang server lifecycle object.

        Raises:
            ValueError: If base_url or timeout values are invalid.
        """
        base_url_value = base_url.strip()
        if not base_url_value:
            raise ValueError("base_url must not be empty.")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0.")

        super().__init__(
            name=name,
            kind="sglang_local",
            default_model=default_model,
            base_url=base_url_value,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._request_timeout_seconds = request_timeout_seconds
        self._managed_server = managed_server

    def capabilities(self) -> BackendCapabilities:
        """Return capability metadata for this backend.

        Returns:
            Capability values for streaming, tool calls, and JSON handling.
        """
        return _SGLANG_CAPABILITIES

    def healthcheck(self) -> BackendStatus:
        """Return static healthcheck status for configured backend.

        Returns:
            Healthy status value for configured SGLang backend.
        """
        return BackendStatus(ok=True, message="SGLang local backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one non-streaming completion.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Normalized completion response.
        """
        self._ensure_server_ready()
        payload = self._build_payload(request)
        response = _post_json_with_retry(
            self._chat_url,
            payload,
            timeout_seconds=self._request_timeout_seconds,
            max_retries=self.max_retries,
        )
        return _parse_completion_response(response, request, provider=self.name)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream completion deltas for one request.

        Args:
            request: Provider-neutral request payload.

        Yields:
            Incremental text, tool-call, and usage deltas.
        """
        self._ensure_server_ready()
        payload = self._build_payload(request)
        payload["stream"] = True
        response = _post_stream_with_retry(
            self._chat_url,
            payload,
            timeout_seconds=self._request_timeout_seconds,
            max_retries=self.max_retries,
        )
        for data in _iter_sse_events(response):
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if not isinstance(choices, list) or not choices:
                continue
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = first_choice.get("delta") or {}
            if not isinstance(delta, dict):
                delta = {}
            content = delta.get("content")
            if content:
                yield LLMDelta(text_delta=str(content))
            for tool_delta in _extract_tool_call_deltas(delta.get("tool_calls")):
                yield LLMDelta(tool_call_delta=tool_delta)
            usage = parse_usage(chunk.get("usage"))
            if usage:
                yield LLMDelta(usage_delta=usage)

    @property
    def _chat_url(self) -> str:
        """Return resolved chat-completions endpoint URL.

        Returns:
            Full URL for chat completions requests.
        """
        base = self.base_url or ""
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        if base.endswith("/v1/"):
            return f"{base}chat/completions"
        if base.endswith("/"):
            return f"{base}v1/chat/completions"
        return f"{base}/v1/chat/completions"

    def _ensure_server_ready(self) -> None:
        """Start managed server if this backend owns one."""
        if self._managed_server is not None:
            self._managed_server.start()

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """Build provider payload from one normalized request.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Request payload suitable for SGLang OpenAI-compatible API.
        """
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _format_messages(request.messages),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        payload.update(request.provider_options)
        return payload


def _post_json_with_retry(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    """POST JSON payload with retry/backoff on retryable errors.

    Args:
        url: Destination URL.
        payload: JSON-serializable request payload.
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        Parsed JSON response object.

    Raises:
        LLMProviderError: If request fails with non-retryable provider errors.
    """
    backoff_seconds = 0.5
    for attempt in range(max_retries + 1):
        try:
            return _post_json(url, payload, timeout_seconds=timeout_seconds)
        except Exception as exc:
            if attempt >= max_retries or not _should_retry(exc):
                raise
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, 8.0)
    return _post_json(url, payload, timeout_seconds=timeout_seconds)


def _post_stream_with_retry(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    max_retries: int,
) -> HTTPResponse:
    """POST streaming payload with retry/backoff on retryable errors.

    Args:
        url: Destination URL.
        payload: JSON-serializable request payload.
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum retry attempts.

    Returns:
        HTTP response stream object for SSE iteration.

    Raises:
        LLMProviderError: If request fails with non-retryable provider errors.
    """
    backoff_seconds = 0.5
    for attempt in range(max_retries + 1):
        try:
            return _post_stream(url, payload, timeout_seconds=timeout_seconds)
        except Exception as exc:
            if attempt >= max_retries or not _should_retry(exc):
                raise
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, 8.0)
    return _post_stream(url, payload, timeout_seconds=timeout_seconds)


def _should_retry(exc: Exception) -> bool:
    """Return whether one exception should trigger retry behavior.

    Args:
        exc: Raised exception from transport or provider layers.

    Returns:
        ``True`` when retry/backoff should be attempted.
    """
    return isinstance(exc, (LLMRateLimitError, LLMProviderError))


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST JSON and parse a JSON-object response.

    Args:
        url: Destination URL.
        payload: JSON-serializable request payload.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Parsed response payload as a JSON object.

    Raises:
        LLMInvalidRequestError: If response format is invalid.
    """
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise LLMInvalidRequestError("SGLang response must be a JSON object.")
            return parsed
    except HTTPError as exc:
        raise _http_error(exc) from exc
    except URLError as exc:
        raise LLMProviderError(str(exc)) from exc


def _post_stream(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> HTTPResponse:
    """POST streaming request and return HTTP response iterator.

    Args:
        url: Destination URL.
        payload: JSON-serializable request payload.
        timeout_seconds: Request timeout in seconds.

    Returns:
        HTTP response object containing SSE chunks.

    Raises:
        LLMProviderError: If transport or provider errors occur.
    """
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return cast(HTTPResponse, urlopen(request, timeout=timeout_seconds))
    except HTTPError as exc:
        raise _http_error(exc) from exc
    except URLError as exc:
        raise LLMProviderError(str(exc)) from exc


def _iter_sse_events(response: Iterable[bytes]) -> Iterator[str]:
    """Iterate decoded SSE ``data:`` payload chunks from one response.

    Args:
        response: Iterable of raw bytes lines from streaming response.

    Yields:
        Decoded SSE payload text for each event.
    """
    buffer: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line:
            if buffer:
                yield "".join(buffer)
                buffer = []
            continue
        if line.startswith("data:"):
            buffer.append(line[len("data:") :].strip())
    if buffer:
        yield "".join(buffer)


def _parse_completion_response(
    response: dict[str, Any],
    request: LLMRequest,
    *,
    provider: str,
) -> LLMResponse:
    """Parse non-streaming completion payload into normalized response.

    Args:
        response: Raw SGLang OpenAI-compatible response payload.
        request: Original request payload.
        provider: Provider/backend name for response metadata.

    Returns:
        Normalized non-streaming LLM response.

    Raises:
        LLMInvalidRequestError: If response choices are missing.
    """
    choices = response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise LLMInvalidRequestError("SGLang response has no choices.")
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    content = message.get("content") or ""
    usage = parse_usage(response.get("usage"))
    return LLMResponse(
        text=str(content).strip(),
        tool_calls=parse_tool_calls(message.get("tool_calls")),
        usage=usage,
        raw=response,
        model=request.model,
        provider=provider,
        finish_reason=first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
    )


def _format_messages(messages: Sequence[object]) -> list[dict[str, Any]]:
    """Serialize provider-neutral messages into OpenAI-compatible payloads.

    Args:
        messages: Sequence of message-like objects.

    Returns:
        Serialized message dictionaries.
    """
    payloads: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role is None or content is None:
            continue
        payload: dict[str, Any] = {"role": role, "content": content}
        name = getattr(message, "name", None)
        if name:
            payload["name"] = name
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        payloads.append(payload)
    return payloads


def _http_error(exc: HTTPError) -> Exception:
    """Translate one HTTP error into a normalized LLM exception.

    Args:
        exc: HTTP transport error raised by ``urlopen``.

    Returns:
        Mapped normalized LLM exception instance.
    """
    message = exc.reason if isinstance(exc.reason, str) else str(exc)
    try:
        body = exc.read().decode("utf-8")
        payload = json.loads(body)
        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                payload_message = error_payload.get("message")
                if isinstance(payload_message, str) and payload_message.strip():
                    message = payload_message.strip()
    except Exception:
        pass

    if exc.code in {401, 403}:
        return LLMAuthError(message)
    if exc.code == 429:
        return LLMRateLimitError(message)
    if exc.code in {400, 404, 422}:
        return LLMInvalidRequestError(message)
    return LLMProviderError(message)


def _extract_tool_call_deltas(raw: Any) -> list[ToolCallDelta]:
    """Extract normalized streamed tool-call deltas from one payload value.

    Args:
        raw: Raw ``tool_calls`` value from a streaming chunk.

    Returns:
        Parsed tool-call delta objects.
    """
    if not isinstance(raw, list):
        return []
    deltas: list[ToolCallDelta] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        call_id = item.get("id")
        deltas.append(
            ToolCallDelta(
                call_id=str(call_id) if call_id else None,
                name=function.get("name") if isinstance(function.get("name"), str) else None,
                arguments_json_delta=(
                    function.get("arguments")
                    if isinstance(function.get("arguments"), str)
                    else None
                ),
            )
        )
    return deltas
