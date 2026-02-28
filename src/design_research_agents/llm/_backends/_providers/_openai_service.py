"""OpenAI service backend using the official OpenAI API."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any

from design_research_agents._contracts._llm import (
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
from design_research_agents._contracts._tools import ToolSpec
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm._backends._errors import map_backend_exception
from design_research_agents.llm._backends._utils import (
    parse_tool_calls,
    parse_usage,
)
from design_research_agents.llm._structured_output import generate_json


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
            name: Stable backend name used in traces and provenance.
            default_model: Default model id for requests without an explicit model.
            api_key_env: Environment variable used as the fallback API-key source.
            api_key: Optional API key override supplied directly by the caller.
            base_url: Optional alternate API base URL for compatible deployments.
            capabilities: Optional capability override for nonstandard model behavior.
            config_hash: Deterministic hash of backend settings.
            max_retries: Maximum retry count for structured fallbacks and provider retries.
            model_patterns: Optional model allowlist patterns accepted by this backend.
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
            Declared capability flags for request validation.
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
            Ready status for a configured backend wrapper.
        """
        return BackendStatus(ok=True, message="OpenAI backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one completion using OpenAI's chat-completions API.

        Args:
            request: Validated request to send to OpenAI.

        Returns:
            Normalized completion response.

        Raises:
            Exception: Propagated after provider errors are normalized.
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
            # Structured-output fallback is only attempted for response-format/schema compatibility failures.
            if _is_response_format_error(mapped_error) and (request.response_schema or request.response_format):
                return self._fallback_prompt_validate(request)
            raise mapped_error from exc

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream completion deltas from OpenAI's chat-completions API.

        Args:
            request: Validated request to stream from OpenAI.

        Yields:
            Normalized text, tool-call, and usage deltas.

        Raises:
            Exception: Propagated after provider errors are normalized.
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
                            arguments_json_delta=(str(arguments_delta) if arguments_delta else None),
                        )
                    )
            usage = getattr(chunk, "usage", None)
            usage_payload = parse_usage(_usage_to_dict(usage))
            if usage_payload:
                yield LLMDelta(usage_delta=usage_payload)

    def _fallback_prompt_validate(self, request: LLMRequest) -> LLMResponse:
        """Fallback to prompt-and-validate JSON generation when native JSON mode fails.

        Args:
            request: Structured-output request that failed native formatting.

        Returns:
            Structured response merged back into the standard response shape.
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
        """Generate once without a ``response_format`` hint.

        Args:
            request: Request to send after stripping native JSON hints.

        Returns:
            Normalized completion response.
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
        """Build the SDK payload for one OpenAI chat-completions request.

        Args:
            request: Request to translate into OpenAI SDK parameters.
            include_response_format: Whether native JSON response hints should be included.

        Returns:
            Keyword-argument payload passed to ``chat.completions.create``.
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
        # Provider options are merged last so caller-specified flags can override defaults.
        request_payload.update(request.provider_options)
        return request_payload

    def _call_with_retry(self, request_payload: dict[str, Any]) -> Any:
        """Call the OpenAI SDK with bounded retry behavior.

        Args:
            request_payload: Payload passed to the OpenAI SDK.

        Returns:
            Raw SDK response object or stream handle.

        Raises:
            Exception: Propagated after retry-eligible failures are exhausted.
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
                # Exponential backoff reduces hot-loop retries on transient provider throttling.
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        return client.chat.completions.create(**request_payload)

    def _create_client(self) -> Any:
        """Instantiate and cache the OpenAI SDK client.

        Returns:
            Configured OpenAI client instance.

        Raises:
            RuntimeError: If the SDK is missing.
        """
        api_key = self._resolve_api_key()
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for openai_service backends. Install with: pip install -e ."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _resolve_api_key(self) -> str:
        """Resolve the API key from direct config or environment.

        Returns:
            Non-empty API key string.

        Raises:
            RuntimeError: If no API key can be resolved.
        """
        if self._api_key:
            return self._api_key
        env_value = os.getenv(self._api_key_env)
        if env_value:
            return env_value
        raise RuntimeError(f"{self._api_key_env} is not set.")


def _format_messages(messages: Sequence[object]) -> list[dict[str, Any]]:
    """Translate normalized messages into OpenAI chat payload entries.

    Args:
        messages: Provider-neutral message objects.

    Returns:
        OpenAI-compatible message payload list.
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
    """Translate one tool spec into OpenAI's function-tool format.

    Args:
        tool: Tool specification exposed by the runtime.

    Returns:
        OpenAI-compatible tool descriptor.
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
    """Translate structured-output hints into OpenAI ``response_format`` payloads.

    Args:
        request: Request containing response-format or schema hints.

    Returns:
        OpenAI ``response_format`` mapping, or ``None`` when not requested.
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
    """Normalize one OpenAI SDK completion response.

    Args:
        completion_response: Raw response object returned by the SDK.
        request: Original request used to produce the response.
        provider: Provider name to stamp into the normalized response.

    Returns:
        Response converted into the shared ``LLMResponse`` contract.

    Raises:
        LLMInvalidRequestError: If the SDK response is missing completion choices.
    """
    choices = getattr(completion_response, "choices", None) or []
    if not choices:
        raise LLMInvalidRequestError("OpenAI response has no choices.")
    response_message = getattr(choices[0], "message", None)
    message_content = getattr(response_message, "content", None) if response_message else None
    response_text = str(message_content or "").strip()
    tool_calls: tuple[ToolCall, ...] = ()
    if response_message is not None:
        tool_calls = parse_tool_calls(_tool_calls_to_list(getattr(response_message, "tool_calls", None)))
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
    """Best-effort conversion of an SDK response into a serializable mapping.

    Args:
        completion_response: Raw OpenAI SDK response object.

    Returns:
        Serializable mapping suitable for ``LLMResponse.raw``.
    """
    try:
        response_payload = completion_response.model_dump()
        if isinstance(response_payload, dict):
            return response_payload
        return {"raw": str(response_payload)}
    except Exception:
        return {"raw": str(completion_response)}


def _tool_calls_to_list(raw_tool_calls: Any) -> list[dict[str, Any]] | None:
    """Convert raw OpenAI tool-call payloads into plain dictionaries.

    Args:
        raw_tool_calls: Tool-call payload from the SDK response.

    Returns:
        Plain-dict tool-call entries, or ``None`` when no usable payload exists.
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
    """Convert raw OpenAI usage payloads into a plain dictionary.

    Args:
        raw_usage: Usage payload from the SDK response.

    Returns:
        Plain usage mapping, or ``None`` when conversion is not possible.
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
    """Return whether an error appears tied to native JSON response formatting.

    Args:
        error: Normalized backend exception to inspect.

    Returns:
        ``True`` when the error message suggests ``response_format`` incompatibility.
    """
    message = str(error).lower()
    return "response_format" in message or "json_schema" in message


def _should_retry(error: Exception) -> bool:
    """Return whether a normalized OpenAI error is retryable.

    Args:
        error: Normalized backend exception to inspect.

    Returns:
        ``True`` for transient provider or rate-limit failures.
    """
    return isinstance(error, (LLMRateLimitError, LLMProviderError))


def _merge_structured_response(structured_output_result: Any) -> LLMResponse:
    """Merge prompt-and-validate output back into a standard response object.

    Args:
        structured_output_result: Result returned by ``generate_json``.

    Returns:
        ``LLMResponse`` containing both normalized text and structured metadata.
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
        text=(structured_output_result.response.text if structured_output_result.response.text else parsed_text),
        tool_calls=structured_output_result.response.tool_calls,
        usage=structured_output_result.response.usage,
        raw=raw_response,
        model=structured_output_result.response.model,
        provider=structured_output_result.response.provider,
        finish_reason=structured_output_result.response.finish_reason,
        latency_ms=structured_output_result.response.latency_ms,
    )
