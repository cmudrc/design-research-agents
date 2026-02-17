"""OpenAI service backend using the official OpenAI API."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolCallDelta,
)
from design_research_agents.contracts.tools import ToolSpec
from design_research_agents.llm.backends.base import BaseLLMBackend
from design_research_agents.llm.backends.errors import map_backend_exception
from design_research_agents.llm.backends.utils import (
    parse_tool_calls,
    parse_usage,
)
from design_research_agents.llm.structured_output import generate_json


class OpenAIServiceBackend(BaseLLMBackend):
    """Backend that calls the OpenAI API via the official SDK."""

    def __init__(
        self,
        *,
        name: str,
        default_model: str,
        api_key_env: str,
        api_key: str | None,
        base_url: str | None,
        capabilities: BackendCapabilities | None = None,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
    ) -> None:
        """Configure OpenAI client defaults and optional capability overrides.

        Args:
            name: Parameter value.
            default_model: Parameter value.
            api_key_env: Parameter value.
            api_key: Parameter value.
            base_url: Parameter value.
            capabilities: Parameter value.
            config_hash: Parameter value.
            max_retries: Parameter value.
            model_patterns: Parameter value.
        """
        super().__init__(
            name=name,
            kind="openai_service",
            default_model=default_model,
            base_url=base_url,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._api_key_env = api_key_env
        self._api_key = api_key
        self._client: Any | None = None
        self._capabilities_override = capabilities

    def capabilities(self) -> BackendCapabilities:
        """Return effective capabilities for this OpenAI backend.

        Returns:
            The resulting value.
        """
        default_caps = BackendCapabilities(
            streaming=True,
            tool_calling="native",
            json_mode="native",
            vision=False,
            max_context_tokens=None,
        )
        return self._capabilities_override or default_caps

    def healthcheck(self) -> BackendStatus:
        """Return static status for a configured OpenAI backend.

        Returns:
            The resulting value.
        """
        return BackendStatus(ok=True, message="OpenAI backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Run generate.

        Args:
            request: Parameter value.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
        request_payload = self._build_payload(request, include_response_format=True)
        try:
            completion_response = self._call_with_retry(request_payload)
            return _parse_completion_response(
                completion_response,
                request,
                provider=self.name,
            )
        except Exception as exc:
            mapped_error = map_backend_exception(exc)
            if _is_response_format_error(mapped_error) and (
                request.response_schema or request.response_format
            ):
                return self._fallback_prompt_validate(request)
            raise mapped_error from exc

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Run stream.

        Args:
            request: Parameter value.

        Yields:
            The yielded values.

        Raises:
            Exception: Raised when execution fails.
        """
        stream_payload = self._build_payload(request, include_response_format=True)
        stream_payload["stream"] = True
        stream_payload["stream_options"] = {"include_usage": True}
        try:
            stream = self._call_with_retry(stream_payload)
        except Exception as exc:
            raise map_backend_exception(exc) from exc

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield LLMDelta(text_delta=str(content))
            tool_calls = getattr(delta, "tool_calls", None)
            if isinstance(tool_calls, list):
                for tool_delta in tool_calls:
                    call_id = getattr(tool_delta, "id", None)
                    function = getattr(tool_delta, "function", None)
                    name = getattr(function, "name", None) if function else None
                    arguments_delta = getattr(function, "arguments", None) if function else None
                    yield LLMDelta(
                        tool_call_delta=ToolCallDelta(
                            call_id=str(call_id) if call_id else None,
                            name=str(name) if name else None,
                            arguments_json_delta=str(arguments_delta) if arguments_delta else None,
                        )
                    )
            usage = getattr(chunk, "usage", None)
            usage_payload = parse_usage(_usage_to_dict(usage))
            if usage_payload:
                yield LLMDelta(usage_delta=usage_payload)

    def _fallback_prompt_validate(self, request: LLMRequest) -> LLMResponse:
        """Run fallback prompt validate.

        Args:
            request: Parameter value.

        Returns:
            The resulting value.
        """
        structured_output_result = generate_json(
            generate_fn=lambda req: self._generate_without_response_format(req),
            request=request,
            schema=request.response_schema,
            max_retries=self.max_retries,
            extra_instructions=None,
        )
        return _merge_structured_response(structured_output_result)

    def _generate_without_response_format(self, request: LLMRequest) -> LLMResponse:
        """Run generate without response format.

        Args:
            request: Parameter value.

        Returns:
            The resulting value.
        """
        request_payload = self._build_payload(request, include_response_format=False)
        completion_response = self._call_with_retry(request_payload)
        return _parse_completion_response(completion_response, request, provider=self.name)

    def _build_payload(
        self,
        request: LLMRequest,
        *,
        include_response_format: bool,
    ) -> dict[str, Any]:
        """Run build payload.

        Args:
            request: Parameter value.
            include_response_format: Parameter value.

        Returns:
            The resulting value.
        """
        request_payload: dict[str, Any] = {
            "model": request.model,
            "messages": _format_messages(request.messages),
        }
        if request.temperature is not None:
            request_payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            request_payload["max_tokens"] = request.max_tokens
        if request.tools:
            request_payload["tools"] = [_format_tool(tool) for tool in request.tools]
        if include_response_format:
            response_format = _format_response_format(request)
            if response_format:
                request_payload["response_format"] = response_format
        request_payload.update(request.provider_options)
        return request_payload

    def _call_with_retry(self, request_payload: dict[str, Any]) -> Any:
        """Run call with retry.

        Args:
            request_payload: Parameter value.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
        client = self._client or self._create_client()
        backoff = 0.5
        for attempt in range(self.max_retries + 1):
            try:
                return client.chat.completions.create(**request_payload)
            except Exception as exc:
                mapped_error = map_backend_exception(exc)
                if attempt >= self.max_retries or not _should_retry(mapped_error):
                    raise mapped_error from exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        return client.chat.completions.create(**request_payload)

    def _create_client(self) -> Any:
        """Run create client.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
        api_key = self._resolve_api_key()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for openai_service backends. "
                "Install with: pip install -e ."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _resolve_api_key(self) -> str:
        """Run resolve api key.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
        if self._api_key:
            return self._api_key
        env_value = os.getenv(self._api_key_env)
        if env_value:
            return env_value
        raise RuntimeError(f"{self._api_key_env} is not set.")


def _format_messages(messages: Sequence[object]) -> list[dict[str, Any]]:
    """Run format messages.

    Args:
        messages: Parameter value.

    Returns:
        The resulting value.
    """
    message_payloads: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role is None or content is None:
            continue
        message_payload: dict[str, Any] = {"role": role, "content": content}
        name = getattr(message, "name", None)
        if name:
            message_payload["name"] = name
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            message_payload["tool_call_id"] = tool_call_id
        message_payloads.append(message_payload)
    return message_payloads


def _format_tool(tool: ToolSpec) -> dict[str, Any]:
    """Run format tool.

    Args:
        tool: Parameter value.

    Returns:
        The resulting value.
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
        request: Parameter value.

    Returns:
        The resulting value.
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


def _parse_completion_response(
    completion_response: Any,
    request: LLMRequest,
    *,
    provider: str,
) -> LLMResponse:
    """Run parse completion response.

    Args:
        completion_response: Parameter value.
        request: Parameter value.
        provider: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    choices = getattr(completion_response, "choices", None) or []
    if not choices:
        raise LLMInvalidRequestError("OpenAI response has no choices.")
    response_message = getattr(choices[0], "message", None)
    message_content = getattr(response_message, "content", None) if response_message else None
    response_text = str(message_content or "").strip()
    tool_calls: tuple[ToolCall, ...] = ()
    if response_message is not None:
        tool_calls = parse_tool_calls(
            _tool_calls_to_list(getattr(response_message, "tool_calls", None))
        )
    usage = parse_usage(_usage_to_dict(getattr(completion_response, "usage", None)))
    return LLMResponse(
        text=response_text,
        tool_calls=tool_calls,
        usage=usage,
        raw=_response_to_dict(completion_response),
        model=request.model,
        provider=provider,
        finish_reason=getattr(choices[0], "finish_reason", None),
    )


def _response_to_dict(completion_response: Any) -> dict[str, Any]:
    """Run response to dict.

    Args:
        completion_response: Parameter value.

    Returns:
        The resulting value.
    """
    try:
        response_payload = completion_response.model_dump()
        if isinstance(response_payload, dict):
            return response_payload
        return {"raw": str(response_payload)}
    except Exception:
        return {"raw": str(completion_response)}


def _tool_calls_to_list(raw_tool_calls: Any) -> list[dict[str, Any]] | None:
    """Run tool calls to list.

    Args:
        raw_tool_calls: Parameter value.

    Returns:
        The resulting value.
    """
    if raw_tool_calls is None:
        return None
    if isinstance(raw_tool_calls, list):
        return [call if isinstance(call, dict) else call.model_dump() for call in raw_tool_calls]
    try:
        return [raw_tool_calls.model_dump()]
    except Exception:
        return None


def _usage_to_dict(raw_usage: Any) -> dict[str, Any] | None:
    """Run usage to dict.

    Args:
        raw_usage: Parameter value.

    Returns:
        The resulting value.
    """
    if raw_usage is None:
        return None
    if isinstance(raw_usage, dict):
        return raw_usage
    dump_fn = getattr(raw_usage, "model_dump", None)
    if not callable(dump_fn):
        return None
    usage_payload = dump_fn()
    if isinstance(usage_payload, dict):
        return usage_payload
    return None


def _is_response_format_error(error: Exception) -> bool:
    """Run is response format error.

    Args:
        error: Parameter value.

    Returns:
        The resulting value.
    """
    message = str(error).lower()
    return "response_format" in message or "json_schema" in message


def _should_retry(error: Exception) -> bool:
    """Run should retry.

    Args:
        error: Parameter value.

    Returns:
        The resulting value.
    """
    return isinstance(error, (LLMRateLimitError, LLMProviderError))


def _merge_structured_response(structured_output_result: Any) -> LLMResponse:
    """Run merge structured response.

    Args:
        structured_output_result: Parameter value.

    Returns:
        The resulting value.
    """
    parsed_text = structured_output_result.parsed
    if not isinstance(parsed_text, str):
        parsed_text = json.dumps(parsed_text, ensure_ascii=True, sort_keys=True)
    raw_response = structured_output_result.response.raw or {}
    raw_response["structured_output"] = {
        "attempts": structured_output_result.attempts + 1,
        "parsed": structured_output_result.parsed,
    }
    return LLMResponse(
        text=(
            structured_output_result.response.text
            if structured_output_result.response.text
            else parsed_text
        ),
        tool_calls=structured_output_result.response.tool_calls,
        usage=structured_output_result.response.usage,
        raw=raw_response,
        model=structured_output_result.response.model,
        provider=structured_output_result.response.provider,
        finish_reason=structured_output_result.response.finish_reason,
        latency_ms=structured_output_result.response.latency_ms,
    )
