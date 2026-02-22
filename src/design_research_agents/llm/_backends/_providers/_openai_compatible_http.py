"""OpenAI-compatible HTTP backend for OpenAI-shaped servers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from http.client import HTTPResponse
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMInvalidRequestError,
    LLMRequest,
    LLMResponse,
    ToolCallDelta,
)
from design_research_agents._contracts._tools import ToolSpec
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm._backends._errors import map_backend_exception
from design_research_agents.llm._backends._utils import (
    parse_tool_calls,
    parse_usage,
)


class OpenAICompatibleHTTPBackend(BaseLLMBackend):
    """Backend that calls any OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        default_model: str,
        api_key_env: str,
        api_key: str | None,
        capabilities: BackendCapabilities,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
    ) -> None:
        """Initialize HTTP endpoint routing and authentication settings.

        Args:
            name: Unique name for this backend configuration.
            base_url: Base URL for the OpenAI-compatible API (e.g. "https://api.example.com/v1").
            default_model: Default model name for prompts that don't specify one.
            api_key_env: Name of the environment variable to read the API key from.
            api_key: Optional API key value to use directly (takes precedence over environment
                variable).
            capabilities: Declared capabilities for this backend (e.g. tool calling and JSON mode
                support levels).
            config_hash: Unique hash of the configuration for caching and invalidation purposes.
            max_retries: Maximum number of retries for generation attempts.
            model_patterns: Optional tuple of glob patterns to match against
                model names for routing purposes.
        """
        super().__init__(
            name=name,
            kind="openai_compatible_http",
            default_model=default_model,
            base_url=base_url,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._api_key_env = api_key_env
        self._api_key = api_key
        self._capabilities = capabilities

    def capabilities(self) -> BackendCapabilities:
        """Return declared capabilities for this endpoint.

        Returns:
            Computed return value.
        """
        return self._capabilities

    def healthcheck(self) -> BackendStatus:
        """Return static status for configured HTTP backend.

        Returns:
            Computed return value.
        """
        return BackendStatus(ok=True, message="OpenAI-compatible backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one completion using the OpenAI-compatible HTTP endpoint.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.
        """
        payload = self._build_payload(request, include_response_format=True)
        response = _post_json(self._chat_url, payload, headers=self._headers())
        return _parse_completion_response(response, request, provider=self.name)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream completion deltas from the OpenAI-compatible endpoint.

        Args:
            request: Input value for this parameter.

        Yields:
            The yielded values.
        """
        payload = self._build_payload(request, include_response_format=True)
        payload["stream"] = True
        response = _post_stream(self._chat_url, payload, headers=self._headers())
        for data in _iter_sse_events(response):
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
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
        """Run chat url.

        Returns:
            Computed return value.
        """
        base = self.base_url or ""
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        if base.endswith("/v1/"):
            return f"{base}chat/completions"
        if base.endswith("/"):
            return f"{base}v1/chat/completions"
        return f"{base}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        """Build HTTP headers for OpenAI-compatible requests.

        Returns:
            Computed return value.
        """
        headers = {"Content-Type": "application/json"}
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _resolve_api_key(self) -> str | None:
        """Run resolve api key.

        Returns:
            Computed return value.
        """
        if self._api_key:
            return self._api_key
        env_value = os.getenv(self._api_key_env)
        return env_value or None

    def _build_payload(
        self,
        request: LLMRequest,
        *,
        include_response_format: bool,
    ) -> dict[str, Any]:
        """Run build payload.

        Args:
            request: Input value for this parameter.
            include_response_format: Input value for this parameter.

        Returns:
            Computed return value.
        """
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _format_messages(request.messages),
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools and self._capabilities.tool_calling == "native":
            payload["tools"] = [_format_tool(tool) for tool in request.tools]
        if include_response_format and self._capabilities.json_mode == "native":
            response_format = _format_response_format(request)
            if response_format:
                payload["response_format"] = response_format
        payload.update(request.provider_options)
        return payload


def _post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str]) -> dict[str, Any]:
    """Run post json.

    Args:
        url: Input value for this parameter.
        payload: Input value for this parameter.
        headers: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=60.0) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                raise LLMInvalidRequestError("OpenAI-compatible response must be a JSON object.")
            return parsed
    except HTTPError as exc:
        raise map_backend_exception(_http_error(exc)) from exc
    except URLError as exc:
        raise map_backend_exception(exc) from exc


def _post_stream(url: str, payload: dict[str, Any], *, headers: dict[str, str]) -> HTTPResponse:
    """Run post stream.

    Args:
        url: Input value for this parameter.
        payload: Input value for this parameter.
        headers: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        return cast(HTTPResponse, urlopen(request, timeout=60.0))
    except HTTPError as exc:
        raise map_backend_exception(_http_error(exc)) from exc
    except URLError as exc:
        raise map_backend_exception(exc) from exc


def _iter_sse_events(response: Iterable[bytes]) -> Iterator[str]:
    """Run iter sse events.

    Args:
        response: Input value for this parameter.

    Yields:
        The yielded values.
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
    """Run parse completion response.

    Args:
        response: Input value for this parameter.
        request: Input value for this parameter.
        provider: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    choices = response.get("choices") or []
    if not choices:
        raise LLMInvalidRequestError("OpenAI-compatible response has no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    tool_calls = parse_tool_calls(message.get("tool_calls"))
    usage = parse_usage(response.get("usage"))
    return LLMResponse(
        text=str(content).strip(),
        tool_calls=tool_calls,
        usage=usage,
        raw=response,
        model=request.model,
        provider=provider,
        finish_reason=choices[0].get("finish_reason"),
    )


def _format_messages(messages: Sequence[object]) -> list[dict[str, Any]]:
    """Run format messages.

    Args:
        messages: Input value for this parameter.

    Returns:
        Computed return value.
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


def _format_tool(tool: ToolSpec) -> dict[str, Any]:
    """Run format tool.

    Args:
        tool: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _format_response_format(request: LLMRequest) -> dict[str, Any] | None:
    """Run format response format.

    Args:
        request: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if request.response_format and isinstance(request.response_format, dict):
        return request.response_format
    if request.response_schema:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": request.response_schema,
            },
        }
    return None


def _http_error(exc: HTTPError) -> Exception:
    """Run http error.

    Args:
        exc: Input value for this parameter.

    Returns:
        Computed return value.
    """
    try:
        body = exc.read().decode("utf-8")
        payload = json.loads(body)
        message = payload.get("error", {}).get("message") or body
        return LLMInvalidRequestError(message)
    except Exception:
        return exc


def _extract_tool_call_deltas(raw: Any) -> list[ToolCallDelta]:
    """Run extract tool call deltas.

    Args:
        raw: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if not isinstance(raw, list):
        return []
    deltas: list[ToolCallDelta] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        call_id = item.get("id")
        function = item.get("function") or {}
        deltas.append(
            ToolCallDelta(
                call_id=str(call_id) if call_id else None,
                name=function.get("name"),
                arguments_json_delta=function.get("arguments"),
            )
        )
    return deltas
