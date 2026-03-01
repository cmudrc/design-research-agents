"""Gemini service backend using the official google-genai SDK."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any, override

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm._backends._errors import map_backend_exception
from design_research_agents.llm._backends._utils import parse_usage
from design_research_agents.llm._structured_output import generate_json


class GeminiServiceBackend(BaseLLMBackend):
    """Backend that calls the Gemini API via the official google-genai SDK."""

    def __init__(
        self,
        *,
        name: str,
        default_model: str,
        api_key_env: str,
        api_key: str | None,
        capabilities: BackendCapabilities | None = None,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
    ) -> None:
        """Configure Gemini client defaults and optional capability overrides."""
        super().__init__(
            name=name,
            kind="gemini_service",
            default_model=default_model,
            base_url=None,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._api_key_env = api_key_env
        self._api_key = api_key
        self._client: Any | None = None
        self._capabilities_override = capabilities

    @override
    def capabilities(self) -> BackendCapabilities:
        """Return effective capabilities for this Gemini backend."""
        default_caps = BackendCapabilities(
            streaming=True,
            tool_calling="none",
            json_mode="native",
            vision=False,
            max_context_tokens=None,
        )
        return self._capabilities_override or default_caps

    @override
    def healthcheck(self) -> BackendStatus:
        """Return static status for a configured Gemini backend."""
        return BackendStatus(ok=True, message="Gemini backend configured.")

    @override
    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one completion using the Gemini generate content API."""
        call_args = self._build_call_args(request, include_response_format=True)
        try:
            generate_response = self._call_with_retry(call_args)
            return _parse_generate_content_response(
                generate_response,
                request,
                provider=self.name,
            )
        except Exception as exc:
            mapped_error = map_backend_exception(_normalize_gemini_exception(exc))
            # Structured-output fallback is only attempted for response-format/schema compatibility failures.
            if _is_response_format_error(mapped_error) and (request.response_schema or request.response_format):
                return self._fallback_prompt_validate(request)
            raise mapped_error from exc

    @override
    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream completion deltas from the Gemini generate content stream API."""
        call_args = self._build_call_args(request, include_response_format=True)
        try:
            stream = self._call_stream_with_retry(call_args)
        except Exception as exc:
            raise map_backend_exception(_normalize_gemini_exception(exc)) from exc

        for chunk in stream:
            text_delta = getattr(chunk, "text", None)
            if isinstance(text_delta, str) and text_delta:
                yield LLMDelta(text_delta=text_delta)
            usage = parse_usage(_usage_metadata_to_dict(getattr(chunk, "usage_metadata", None)))
            if usage:
                yield LLMDelta(usage_delta=usage)

    def _fallback_prompt_validate(self, request: LLMRequest) -> LLMResponse:
        """Prompt+validate fallback for structured output compatibility."""
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
        call_args = self._build_call_args(request, include_response_format=False)
        generate_response = self._call_with_retry(call_args)
        return _parse_generate_content_response(generate_response, request, provider=self.name)

    def _build_call_args(
        self,
        request: LLMRequest,
        *,
        include_response_format: bool,
    ) -> dict[str, Any]:
        """Build one google-genai generate_content call payload."""
        call_args: dict[str, Any] = {
            "model": request.model,
            "contents": _format_contents(request.messages),
        }
        config = _build_generate_config(request, include_response_format=include_response_format)
        if config:
            call_args["config"] = config
        return call_args

    def _call_with_retry(self, call_args: dict[str, Any]) -> Any:
        """Call Gemini generate_content with retry for transient provider failures."""
        client = self._client or self._create_client()
        backoff = 0.5
        for attempt in range(self.max_retries + 1):
            try:
                return client.models.generate_content(**call_args)
            except Exception as exc:
                normalized = _normalize_gemini_exception(exc)
                mapped_error = map_backend_exception(normalized)
                if attempt >= self.max_retries or not _should_retry(mapped_error):
                    raise mapped_error from exc
                # Exponential backoff reduces hot-loop retries on transient provider throttling.
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        return client.models.generate_content(**call_args)

    def _call_stream_with_retry(self, call_args: dict[str, Any]) -> Any:
        """Call Gemini generate_content_stream with retry for transient provider failures."""
        client = self._client or self._create_client()
        backoff = 0.5
        for attempt in range(self.max_retries + 1):
            try:
                return client.models.generate_content_stream(**call_args)
            except Exception as exc:
                normalized = _normalize_gemini_exception(exc)
                mapped_error = map_backend_exception(normalized)
                if attempt >= self.max_retries or not _should_retry(mapped_error):
                    raise mapped_error from exc
                # Exponential backoff reduces hot-loop retries on transient provider throttling.
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        return client.models.generate_content_stream(**call_args)

    def _create_client(self) -> Any:
        """Create and cache the google-genai client instance."""
        api_key = self._resolve_api_key()
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' package is required for gemini_service backends. "
                'Install with: pip install -e ".[gemini]"'
            ) from exc
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _resolve_api_key(self) -> str:
        """Resolve API key from explicit value, primary env var, then Gemini fallback env var."""
        if self._api_key:
            return self._api_key
        env_value = os.getenv(self._api_key_env)
        if env_value:
            return env_value
        if self._api_key_env == "GOOGLE_API_KEY":
            fallback_value = os.getenv("GEMINI_API_KEY")
            if fallback_value:
                return fallback_value
        raise RuntimeError(f"{self._api_key_env} is not set.")


def _format_contents(messages: Sequence[object]) -> list[dict[str, Any]]:
    """Format provider-neutral chat messages to Gemini contents payload."""
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "system" or not isinstance(content, str):
            continue
        gemini_role = "model" if role == "assistant" else "user"
        if role == "tool":
            content = _format_tool_message_content(message, content=content)
        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    if not contents:
        # Preserve a valid request payload shape when caller sends system-only context.
        contents.append({"role": "user", "parts": [{"text": ""}]})
    return contents


def _format_tool_message_content(message: object, *, content: str) -> str:
    """Normalize tool-role message content for Gemini user-role fallback."""
    tool_name = getattr(message, "tool_name", None)
    tool_call_id = getattr(message, "tool_call_id", None)
    header = "tool"
    if isinstance(tool_name, str) and tool_name:
        header = f"{header}[{tool_name}]"
    if isinstance(tool_call_id, str) and tool_call_id:
        header = f"{header}#{tool_call_id}"
    return f"{header}: {content}"


def _extract_system_instruction(messages: Sequence[object]) -> str | None:
    """Extract and join system messages for Gemini config.system_instruction."""
    segments: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role == "system" and isinstance(content, str) and content:
            segments.append(content)
    if not segments:
        return None
    return "\n\n".join(segments)


def _build_generate_config(
    request: LLMRequest,
    *,
    include_response_format: bool,
) -> dict[str, Any]:
    """Build Gemini generate content config payload."""
    config: dict[str, Any] = {}
    system_instruction = _extract_system_instruction(request.messages)
    if system_instruction:
        config["system_instruction"] = system_instruction
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.max_tokens is not None:
        config["max_output_tokens"] = request.max_tokens
    if include_response_format:
        response_config = _format_response_config(request)
        config.update(response_config)
    # Provider options are merged last so caller-specified flags can override defaults.
    config.update(request.provider_options)
    return config


def _format_response_config(request: LLMRequest) -> dict[str, Any]:
    """Build Gemini structured-output config from request schema/format hints."""
    if request.response_schema:
        return {
            "response_mime_type": "application/json",
            "response_schema": request.response_schema,
        }
    response_format = request.response_format
    if not isinstance(response_format, dict):
        return {}

    config: dict[str, Any] = {}
    response_mime_type = response_format.get("response_mime_type")
    if isinstance(response_mime_type, str) and response_mime_type:
        config["response_mime_type"] = response_mime_type

    if response_format.get("type") in {"json_object", "json_schema"}:
        config["response_mime_type"] = "application/json"

    response_schema = response_format.get("response_schema")
    if isinstance(response_schema, dict):
        config["response_schema"] = response_schema
    else:
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            schema = json_schema.get("schema")
            if isinstance(schema, dict):
                config["response_schema"] = schema

    return config


def _parse_generate_content_response(
    generate_response: Any,
    request: LLMRequest,
    *,
    provider: str,
) -> LLMResponse:
    """Parse a Gemini generate-content response into normalized contract shape."""
    response_text = str(getattr(generate_response, "text", "") or "").strip()
    usage = parse_usage(_usage_metadata_to_dict(getattr(generate_response, "usage_metadata", None)))
    finish_reason = _extract_finish_reason(generate_response)
    return LLMResponse(
        text=response_text,
        tool_calls=(),
        usage=usage,
        raw=_response_to_dict(generate_response),
        model=request.model,
        provider=provider,
        finish_reason=finish_reason,
    )


def _extract_finish_reason(generate_response: Any) -> str | None:
    """Extract first-candidate finish reason from Gemini response payload."""
    candidates = getattr(generate_response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return None
    enum_value = getattr(finish_reason, "value", None)
    if isinstance(enum_value, str) and enum_value:
        return enum_value
    finish_reason_text = str(finish_reason).strip()
    return finish_reason_text or None


def _usage_metadata_to_dict(raw_usage: Any) -> dict[str, Any] | None:
    """Convert Gemini usage_metadata payload into OpenAI-style usage fields."""
    payload = _to_dict(raw_usage)
    if not isinstance(payload, dict):
        return None

    prompt_tokens = _coerce_int(payload.get("prompt_token_count"))
    completion_tokens = _coerce_int(payload.get("candidates_token_count"))
    total_tokens = _coerce_int(payload.get("total_token_count"))

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _response_to_dict(generate_response: Any) -> dict[str, Any]:
    """Convert Gemini response object into plain dict for raw payload capture."""
    payload = _to_dict(generate_response)
    if isinstance(payload, dict):
        return payload
    return {"raw": str(generate_response)}


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


def _normalize_gemini_exception(exc: Exception) -> Exception:
    """Attach status_code from Gemini APIError.code for taxonomy mapping."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        try:
            exc.status_code = code  # type: ignore[attr-defined]
        except Exception:
            return exc
    return exc


def _is_response_format_error(error: Exception) -> bool:
    """Return true when an error indicates unsupported JSON/response format hints."""
    message = str(error).lower()
    return (
        "response_format" in message
        or "response_schema" in message
        or "response_mime_type" in message
        or "json_schema" in message
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


def _coerce_int(value: Any) -> int | None:
    """Coerce nullable usage counter fields to int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
