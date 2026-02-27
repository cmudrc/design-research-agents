"""Anthropic service backend using the official anthropic SDK."""

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
from design_research_agents.llm._backends._utils import parse_usage
from design_research_agents.llm._structured_output import generate_json


class AnthropicServiceBackend(BaseLLMBackend):
    """Backend that calls the Anthropic API via the official SDK."""

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
        """Configure Anthropic client defaults and optional capability overrides."""
        super().__init__(
            name=name,
            kind="anthropic_service",
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
        """Return effective capabilities for this Anthropic backend."""
        default_caps = BackendCapabilities(
            streaming=True,
            tool_calling="native",
            json_mode="native",
            vision=False,
            max_context_tokens=None,
        )
        return self._capabilities_override or default_caps

    def healthcheck(self) -> BackendStatus:
        """Return static status for a configured Anthropic backend."""
        return BackendStatus(ok=True, message="Anthropic backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one completion using the Anthropic messages API."""
        request_payload = self._build_payload(request, include_response_format=True)
        try:
            message_response = self._call_with_retry(request_payload)
            return _parse_message_response(
                message_response,
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
        """Stream completion deltas from the Anthropic messages API."""
        stream_payload = self._build_payload(request, include_response_format=True)
        stream_payload["stream"] = True
        try:
            stream = self._call_with_retry(stream_payload)
        except Exception as exc:
            raise map_backend_exception(exc) from exc

        tool_state_by_index: dict[int, tuple[str | None, str | None]] = {}
        for event in stream:
            yield from _stream_event_deltas(event, tool_state_by_index=tool_state_by_index)

    def _fallback_prompt_validate(self, request: LLMRequest) -> LLMResponse:
        """Run prompt+validate fallback for structured output compatibility."""
        structured_output_result = generate_json(
            generate_fn=lambda req: self._generate_without_response_format(req),
            request=request,
            schema=request.response_schema,
            max_retries=self.max_retries,
            extra_instructions=None,
        )
        return _merge_structured_response(structured_output_result)

    def _generate_without_response_format(self, request: LLMRequest) -> LLMResponse:
        """Generate one completion without response-format hints."""
        request_payload = self._build_payload(request, include_response_format=False)
        message_response = self._call_with_retry(request_payload)
        return _parse_message_response(message_response, request, provider=self.name)

    def _build_payload(
        self,
        request: LLMRequest,
        *,
        include_response_format: bool,
    ) -> dict[str, Any]:
        """Build one Anthropic messages request payload."""
        request_payload: dict[str, Any] = {
            "model": request.model,
            "messages": _format_messages(request.messages),
            # Anthropic requires max_tokens on message requests.
            "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
        }
        system_instruction = _extract_system_instruction(request.messages)
        if system_instruction:
            request_payload["system"] = system_instruction
        if request.temperature is not None:
            request_payload["temperature"] = request.temperature
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
        """Call Anthropic messages API with retry for transient provider failures."""
        client = self._client or self._create_client()
        backoff = 0.5
        for attempt in range(self.max_retries + 1):
            try:
                return client.messages.create(**request_payload)
            except Exception as exc:
                mapped_error = map_backend_exception(exc)
                if attempt >= self.max_retries or not _should_retry(mapped_error):
                    raise mapped_error from exc
                # Exponential backoff reduces hot-loop retries on transient provider throttling.
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        return client.messages.create(**request_payload)

    def _create_client(self) -> Any:
        """Create and cache the Anthropic SDK client instance."""
        api_key = self._resolve_api_key()
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required for anthropic_service backends. "
                'Install with: pip install -e ".[anthropic]"'
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = Anthropic(**kwargs)
        return self._client

    def _resolve_api_key(self) -> str:
        """Resolve API key from explicit value or configured environment variable."""
        if self._api_key:
            return self._api_key
        env_value = os.getenv(self._api_key_env)
        if env_value:
            return env_value
        raise RuntimeError(f"{self._api_key_env} is not set.")


def _extract_system_instruction(messages: Sequence[object]) -> str | None:
    """Extract and join system messages for Anthropic system instructions."""
    segments: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "system" and isinstance(content, str) and content:
            segments.append(content)
    if not segments:
        return None
    return "\n\n".join(segments)


def _format_messages(messages: Sequence[object]) -> list[dict[str, Any]]:
    """Format provider-neutral chat messages for Anthropic message payloads."""
    message_payloads: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "system" or not isinstance(content, str):
            continue
        if role == "assistant":
            message_payloads.append({"role": "assistant", "content": content})
            continue
        if role == "tool":
            message_payloads.append(_format_tool_result_message(message, content=content))
            continue
        message_payloads.append({"role": "user", "content": content})
    if not message_payloads:
        # Preserve a valid request payload shape when caller sends system-only context.
        message_payloads.append({"role": "user", "content": ""})
    return message_payloads


def _format_tool_result_message(message: object, *, content: str) -> dict[str, Any]:
    """Format one tool result message for Anthropic tool-use continuation."""
    tool_use_id = getattr(message, "tool_call_id", None)
    if isinstance(tool_use_id, str) and tool_use_id:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ],
        }
    return {
        "role": "user",
        "content": _format_tool_message_content(message, content=content),
    }


def _format_tool_message_content(message: object, *, content: str) -> str:
    """Normalize tool-role message content for text-only fallback."""
    tool_name = getattr(message, "tool_name", None)
    tool_call_id = getattr(message, "tool_call_id", None)
    header = "tool"
    if isinstance(tool_name, str) and tool_name:
        header = f"{header}[{tool_name}]"
    if isinstance(tool_call_id, str) and tool_call_id:
        header = f"{header}#{tool_call_id}"
    return f"{header}: {content}"


def _format_tool(tool: ToolSpec) -> dict[str, Any]:
    """Format one tool specification for Anthropic payloads."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _format_response_format(request: LLMRequest) -> dict[str, Any] | None:
    """Format response format payload from explicit response_format or schema hints."""
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


def _parse_message_response(
    message_response: Any,
    request: LLMRequest,
    *,
    provider: str,
) -> LLMResponse:
    """Parse one Anthropic message response into normalized contract payload."""
    content = _get_field(message_response, "content")
    response_text, tool_calls = _extract_text_and_tool_calls(content)
    usage = parse_usage(_usage_to_dict(_get_field(message_response, "usage")))
    model = _optional_str(_get_field(message_response, "model")) or request.model
    finish_reason = _optional_str(_get_field(message_response, "stop_reason"))
    return LLMResponse(
        text=response_text,
        tool_calls=tool_calls,
        usage=usage,
        raw=_response_to_dict(message_response),
        model=model,
        provider=provider,
        finish_reason=finish_reason,
    )


def _extract_text_and_tool_calls(content_blocks: Any) -> tuple[str, tuple[ToolCall, ...]]:
    """Extract text and tool-call blocks from Anthropic content payload."""
    if not isinstance(content_blocks, list):
        return "", ()

    text_segments: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content_blocks:
        block_type = _optional_str(_get_field(block, "type"))
        if block_type == "text":
            text = _optional_str(_get_field(block, "text"))
            if text:
                text_segments.append(text)
            continue

        if block_type != "tool_use":
            continue

        tool_name = _optional_str(_get_field(block, "name"))
        if not tool_name:
            continue
        call_id = _optional_str(_get_field(block, "id")) or f"call_{len(tool_calls) + 1}"
        arguments_json = _serialize_tool_input(_get_field(block, "input"))
        tool_calls.append(
            ToolCall(
                name=tool_name,
                arguments_json=arguments_json,
                call_id=call_id,
            )
        )

    return "".join(text_segments).strip(), tuple(tool_calls)


def _stream_event_deltas(
    event: Any,
    *,
    tool_state_by_index: dict[int, tuple[str | None, str | None]],
) -> Iterator[LLMDelta]:
    """Yield zero or more normalized deltas for one Anthropic stream event."""
    event_type = _optional_str(_get_field(event, "type"))

    usage_delta = _usage_delta_from_stream_event(event, event_type=event_type)
    if usage_delta is not None:
        yield usage_delta

    if event_type == "content_block_start":
        _capture_tool_use_start_event(event, tool_state_by_index=tool_state_by_index)
        return

    if event_type != "content_block_delta":
        return

    text_delta = _text_delta_from_stream_event(event)
    if text_delta is not None:
        yield text_delta
        return

    tool_call_delta = _tool_call_delta_from_stream_event(event, tool_state_by_index=tool_state_by_index)
    if tool_call_delta is not None:
        yield tool_call_delta


def _usage_delta_from_stream_event(event: Any, *, event_type: str | None) -> LLMDelta | None:
    """Extract usage deltas from message_start/message_delta stream events."""
    usage_raw: Any = None
    if event_type == "message_start":
        message_payload = _get_field(event, "message")
        usage_raw = _get_field(message_payload, "usage")
    elif event_type == "message_delta":
        usage_raw = _get_field(event, "usage")
    usage_payload = parse_usage(_usage_to_dict(usage_raw))
    if usage_payload is None:
        return None
    return LLMDelta(usage_delta=usage_payload)


def _capture_tool_use_start_event(
    event: Any,
    *,
    tool_state_by_index: dict[int, tuple[str | None, str | None]],
) -> None:
    """Capture tool-use block metadata so later json deltas have call id + name context."""
    index = _coerce_int(_get_field(event, "index"))
    content_block = _get_field(event, "content_block")
    if index is None or _optional_str(_get_field(content_block, "type")) != "tool_use":
        return
    tool_state_by_index[index] = (
        _optional_str(_get_field(content_block, "id")),
        _optional_str(_get_field(content_block, "name")),
    )


def _text_delta_from_stream_event(event: Any) -> LLMDelta | None:
    """Extract one text delta from content_block_delta events when present."""
    delta_payload = _get_field(event, "delta")
    if _optional_str(_get_field(delta_payload, "type")) != "text_delta":
        return None
    text_delta = _optional_str(_get_field(delta_payload, "text"))
    if not text_delta:
        return None
    return LLMDelta(text_delta=text_delta)


def _tool_call_delta_from_stream_event(
    event: Any,
    *,
    tool_state_by_index: dict[int, tuple[str | None, str | None]],
) -> LLMDelta | None:
    """Extract one tool-call delta from input_json_delta stream events when present."""
    delta_payload = _get_field(event, "delta")
    if _optional_str(_get_field(delta_payload, "type")) != "input_json_delta":
        return None
    arguments_delta = _optional_str(_get_field(delta_payload, "partial_json"))
    if not arguments_delta:
        arguments_delta = _optional_str(_get_field(delta_payload, "text"))
    if not arguments_delta:
        return None
    index = _coerce_int(_get_field(event, "index"))
    call_id: str | None = None
    name: str | None = None
    if index is not None:
        call_id, name = tool_state_by_index.get(index, (None, None))
    return LLMDelta(
        tool_call_delta=ToolCallDelta(
            call_id=call_id,
            name=name,
            arguments_json_delta=arguments_delta,
        )
    )


def _serialize_tool_input(raw_input: Any) -> str:
    """Serialize Anthropic tool-use input payload into JSON text."""
    if isinstance(raw_input, str):
        return raw_input
    if isinstance(raw_input, dict):
        return json.dumps(raw_input, ensure_ascii=True, sort_keys=True)
    if raw_input is None:
        return "{}"
    try:
        return json.dumps(raw_input, ensure_ascii=True, sort_keys=True)
    except Exception:
        return str(raw_input)


def _response_to_dict(message_response: Any) -> dict[str, Any]:
    """Convert response payload to dict for normalized raw capture."""
    try:
        response_payload = _to_dict(message_response)
        if isinstance(response_payload, dict):
            return response_payload
        return {"raw": str(message_response)}
    except Exception:
        return {"raw": str(message_response)}


def _to_dict(value: Any) -> dict[str, Any] | None:
    """Convert SDK model objects to plain dict when dump helpers are available."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, dict):
            return dumped
    return None


def _usage_to_dict(raw_usage: Any) -> dict[str, Any] | None:
    """Convert Anthropic usage objects to OpenAI-style usage fields."""
    payload = _to_dict(raw_usage)
    if not isinstance(payload, dict):
        return None

    prompt_tokens = _coerce_int(payload.get("input_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _coerce_int(payload.get("prompt_tokens"))

    completion_tokens = _coerce_int(payload.get("output_tokens"))
    if completion_tokens is None:
        completion_tokens = _coerce_int(payload.get("completion_tokens"))

    total_tokens = _coerce_int(payload.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _is_response_format_error(error: Exception) -> bool:
    """Return true when an error indicates unsupported JSON/response format hints."""
    message = str(error).lower()
    return (
        "response_format" in message
        or "response schema" in message
        or "response_schema" in message
        or "json_schema" in message
        or "json mode" in message
    )


def _should_retry(error: Exception) -> bool:
    """Return true when one error is retryable."""
    return isinstance(error, (LLMRateLimitError, LLMProviderError))


def _merge_structured_response(structured_output_result: Any) -> LLMResponse:
    """Merge structured-output fallback metadata into normalized response payload."""
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


def _get_field(value: Any, key: str) -> Any:
    """Read field values from dict-like or attribute-based SDK objects."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _optional_str(value: Any) -> str | None:
    """Convert arbitrary values to non-empty strings when possible."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    """Coerce nullable usage counter fields to int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
