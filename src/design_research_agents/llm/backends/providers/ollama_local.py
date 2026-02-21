"""Ollama local backend using native ``/api/chat`` endpoints."""

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
    Usage,
)
from design_research_agents.llm.backends.base import BaseLLMBackend
from design_research_agents.llm.backends.utils import (
    parse_tool_calls,
)

from .ollama_server import OllamaServerBackend

_OLLAMA_CAPABILITIES = BackendCapabilities(
    streaming=True,
    tool_calling="best_effort",
    json_mode="prompt+validate",
    vision=False,
    max_context_tokens=None,
)


class OllamaLocalBackend(BaseLLMBackend):
    """Backend for local/self-hosted Ollama chat APIs."""

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
        managed_server: OllamaServerBackend | None = None,
    ) -> None:
        """Initialize Ollama backend transport and optional managed server.

        Args:
            name: Unique backend name for tracing and routing.
            base_url: Base URL for the Ollama API endpoint.
            default_model: Default model id used when request model is omitted.
            request_timeout_seconds: HTTP request timeout in seconds.
            config_hash: Stable hash of backend config inputs.
            max_retries: Maximum retry attempts for generation requests.
            model_patterns: Optional glob-like model patterns for selector routing.
            managed_server: Optional managed Ollama daemon lifecycle object.

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
            kind="ollama_local",
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
        return _OLLAMA_CAPABILITIES

    def healthcheck(self) -> BackendStatus:
        """Return static healthcheck status for configured backend.

        Returns:
            Healthy status value for configured Ollama backend.
        """
        return BackendStatus(ok=True, message="Ollama local backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one non-streaming completion.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Normalized completion response.
        """
        self._ensure_server_ready()
        payload = self._build_payload(request, stream=False)
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
        payload = self._build_payload(request, stream=True)
        response = _post_stream_with_retry(
            self._chat_url,
            payload,
            timeout_seconds=self._request_timeout_seconds,
            max_retries=self.max_retries,
        )
        for chunk in _iter_json_events(response):
            message = chunk.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    yield LLMDelta(text_delta=content)
                for tool_delta in _extract_tool_call_deltas(message.get("tool_calls")):
                    yield LLMDelta(tool_call_delta=tool_delta)
            if bool(chunk.get("done")):
                usage = _parse_ollama_usage(chunk)
                if usage is not None:
                    yield LLMDelta(usage_delta=usage)
                break

    @property
    def _chat_url(self) -> str:
        """Return resolved Ollama chat endpoint URL.

        Returns:
            Full URL for ``/api/chat`` requests.
        """
        base = (self.base_url or "").rstrip("/")
        return f"{base}/api/chat"

    def _ensure_server_ready(self) -> None:
        """Start managed server if this backend owns one."""
        if self._managed_server is not None:
            self._managed_server.start()

    def _build_payload(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        """Build Ollama payload from one normalized request.

        Args:
            request: Provider-neutral request payload.
            stream: Whether streaming responses are requested.

        Returns:
            Request payload suitable for the Ollama chat API.
        """
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _format_messages(request.messages),
            "stream": stream,
        }
        options: dict[str, object] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        provider_options = dict(request.provider_options)
        provider_options_options = provider_options.pop("options", None)
        if isinstance(provider_options_options, dict):
            options.update(provider_options_options)
        if options:
            payload["options"] = options
        payload.update(provider_options)
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
        HTTP response stream object for chunk iteration.

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
                raise LLMInvalidRequestError("Ollama response must be a JSON object.")
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
        HTTP response object containing streamed JSON chunks.

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


def _iter_json_events(response: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Iterate JSON events from Ollama streaming responses.

    Args:
        response: Iterable of raw bytes lines from streaming response.

    Yields:
        Parsed JSON dictionary events.
    """
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _parse_completion_response(
    response: dict[str, Any],
    request: LLMRequest,
    *,
    provider: str,
) -> LLMResponse:
    """Parse non-streaming completion payload into normalized response.

    Args:
        response: Raw Ollama chat response payload.
        request: Original request payload.
        provider: Provider/backend name for response metadata.

    Returns:
        Normalized non-streaming LLM response.
    """
    message = response.get("message")
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    usage = _parse_ollama_usage(response)
    return LLMResponse(
        text=str(content).strip() if isinstance(content, str) else "",
        tool_calls=parse_tool_calls(message.get("tool_calls")),
        usage=usage,
        raw=response,
        model=request.model,
        provider=provider,
        finish_reason=(
            response.get("done_reason") if isinstance(response.get("done_reason"), str) else None
        ),
    )


def _parse_ollama_usage(payload: dict[str, Any]) -> Usage | None:
    """Parse Ollama token accounting fields into normalized usage.

    Args:
        payload: Raw Ollama response payload.

    Returns:
        Normalized usage counters when available.
    """
    prompt_tokens = _coerce_int(payload.get("prompt_eval_count"))
    completion_tokens = _coerce_int(payload.get("eval_count"))
    total_tokens: int | None = None
    if prompt_tokens is not None or completion_tokens is not None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _coerce_int(value: Any) -> int | None:
    """Coerce one numeric-like value into an integer.

    Args:
        value: Value to coerce.

    Returns:
        Integer value when coercion is possible, otherwise ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _format_messages(messages: Sequence[object]) -> list[dict[str, Any]]:
    """Serialize provider-neutral messages into Ollama payloads.

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
            payload_error = payload.get("error")
            if isinstance(payload_error, str) and payload_error.strip():
                message = payload_error.strip()
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
    """Extract normalized tool-call deltas from one streamed payload value.

    Args:
        raw: Raw ``tool_calls`` value from an Ollama message chunk.

    Returns:
        Parsed tool-call delta objects.
    """
    if not isinstance(raw, list):
        return []
    deltas: list[ToolCallDelta] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        function_payload = function if isinstance(function, dict) else {}
        call_id_raw = item.get("id")
        call_id = str(call_id_raw) if call_id_raw else f"call_{index}"
        name = function_payload.get("name")
        args_payload = function_payload.get("arguments")
        args_text: str | None = None
        if isinstance(args_payload, str):
            args_text = args_payload
        elif isinstance(args_payload, dict):
            args_text = json.dumps(args_payload, ensure_ascii=True, sort_keys=True)
        deltas.append(
            ToolCallDelta(
                call_id=call_id,
                name=name if isinstance(name, str) and name else None,
                arguments_json_delta=args_text,
            )
        )
    return deltas
