"""Backend adapters that normalize provider behavior to contract interfaces.

Each adapter encapsulates provider-specific request/response mechanics and
exception mapping so the rest of the package can interact with a unified set of
LLM contracts.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.llm import (
    LLMAuthError,
    LLMChatParams,
    LLMError,
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderAdapter,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMStreamEvent,
)

from .echo_test_backend import complete as echo_test_complete
from .llama_cpp_server import LlamaCppServerBackend
from .openai import complete as openai_complete
from .transformers_backend import TransformersBackend
from .types import BackendName


@dataclass(frozen=True, slots=True)
class OpenAIBackendConfig:
    """Immutable OpenAI configuration payload consumed by provider adapters.

    A value object is used here so request-time adapters cannot accidentally
    mutate global process configuration.
    """

    api_key_env: str
    api_key: str | None
    base_url: str | None
    require_api_key: bool


class EchoTestProviderAdapter(LLMProviderAdapter):
    """Adapter for the deterministic echo-test backend.

    Useful for local verification where deterministic behavior is required.
    """

    provider_name = "echo-test"

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate an echo-test backend response from contract chat inputs.

        Args:
            messages: Ordered chat messages to include in the prompt.
            model: Requested model identifier propagated to response metadata.
            params: Shared generation parameters and provider options.

        Returns:
            Normalized LLM response from the echo-test backend.

        Raises:
            LLMError: If backend execution fails.
        """
        # Measure wall time around provider calls so latency metadata stays comparable.
        start = time.perf_counter()
        prompt = _messages_to_prompt(messages, response_schema=params.response_schema)
        try:
            text = echo_test_complete(prompt)
        except Exception as exc:  # pragma: no cover - defensive mapping for future providers.
            raise _map_backend_exception(exc) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(
            model=model,
            text=text,
            provider=self.provider_name,
            finish_reason="completed",
            latency_ms=latency_ms,
            raw_output={
                "backend": self.provider_name,
                "requested_model": model,
                "response_schema": params.response_schema,
                # Keep provider options for downstream debugging/telemetry.
                "provider_options": dict(params.provider_options),
            },
        )

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream echo-test backend output as normalized events.

        Args:
            messages: Ordered chat messages to include in the prompt.
            model: Requested model identifier propagated to response metadata.
            params: Shared generation parameters and provider options.

        Yields:
            Stream events containing text delta and final completion payload.
        """
        # Echo-test is non-streaming; we synthesize two standard stream events.
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


@dataclass(slots=True)
class OpenAIProviderAdapter(LLMProviderAdapter):
    """Adapter for hosted OpenAI-compatible providers.

    Wraps shared OpenAI configuration and normalizes output/error contracts.
    """

    config: OpenAIBackendConfig
    provider_name: str = "openai"

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate a hosted-provider response from contract chat inputs.

        Args:
            messages: Ordered chat messages to include in the prompt.
            model: Requested model identifier sent to the provider.
            params: Shared generation parameters and provider options.

        Returns:
            Normalized LLM response from the OpenAI-compatible backend.

        Raises:
            LLMError: If provider execution fails.
        """
        # Measure wall time around provider calls so latency metadata stays comparable.
        start = time.perf_counter()
        prompt = _messages_to_prompt(messages, response_schema=params.response_schema)
        try:
            text = openai_complete(
                prompt,
                model=model,
                api_key_env=self.config.api_key_env,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                require_api_key=self.config.require_api_key,
            )
        except Exception as exc:
            raise _map_backend_exception(exc) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(
            model=model,
            text=text,
            provider=self.provider_name,
            finish_reason="completed",
            latency_ms=latency_ms,
            raw_output={
                "backend": self.provider_name,
                "requested_model": model,
                "response_schema": params.response_schema,
                # Keep provider options for downstream debugging/telemetry.
                "provider_options": dict(params.provider_options),
            },
        )

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream hosted-provider output as normalized events.

        Args:
            messages: Ordered chat messages to include in the prompt.
            model: Requested model identifier sent to the provider.
            params: Shared generation parameters and provider options.

        Yields:
            Stream events containing text delta and final completion payload.
        """
        # Current implementation wraps non-streaming call until native streaming is wired.
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


@dataclass(slots=True)
class LlamaCppServerProviderAdapter(LLMProviderAdapter):
    """Adapter for process-managed llama-cpp server backend.

    Delegates completion to the managed local OpenAI-compatible server wrapper.
    """

    backend: LlamaCppServerBackend | None
    provider_name: str = "llama-cpp-server"

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate a llama-cpp-server response from contract chat inputs.

        Args:
            messages: Ordered chat messages to include in the prompt.
            model: Requested model identifier propagated to response metadata.
            params: Shared generation parameters and provider options.

        Returns:
            Normalized LLM response from the managed llama-cpp server backend.

        Raises:
            LLMInvalidRequestError: If llama-cpp server is not configured.
            LLMError: If backend execution fails.
        """
        # Backend is optional at config time, but required for invocation.
        if self.backend is None:
            raise LLMInvalidRequestError(
                "llama-cpp-server backend is not configured. "
                "Call configure_llama_cpp_server(model=...) before use."
            )

        # Measure wall time around provider calls so latency metadata stays comparable.
        start = time.perf_counter()
        prompt = _messages_to_prompt(messages, response_schema=params.response_schema)
        try:
            text = self.backend.complete(prompt)
        except Exception as exc:
            raise _map_backend_exception(exc) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(
            model=model,
            text=text,
            provider=self.provider_name,
            finish_reason="completed",
            latency_ms=latency_ms,
            raw_output={
                "backend": self.provider_name,
                "requested_model": model,
                "response_schema": params.response_schema,
                # Keep provider options for downstream debugging/telemetry.
                "provider_options": dict(params.provider_options),
            },
        )

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream llama-cpp-server output as normalized events.

        Args:
            messages: Ordered chat messages to include in the prompt.
            model: Requested model identifier propagated to response metadata.
            params: Shared generation parameters and provider options.

        Yields:
            Stream events containing text delta and final completion payload.
        """
        # Current implementation wraps non-streaming call until native streaming is wired.
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


@dataclass(slots=True)
class TransformersProviderAdapter(LLMProviderAdapter):
    """Adapter for local Hugging Face Transformers backend."""

    backend: TransformersBackend | None
    provider_name: str = "transformers"

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Generate a Transformers-backed response from contract chat inputs."""
        if self.backend is None:
            raise LLMInvalidRequestError(
                "transformers backend is not configured. "
                "Call configure_transformers(model=...) before use."
            )

        start = time.perf_counter()
        prompt = _messages_to_prompt(messages, response_schema=params.response_schema)
        generation_kwargs = _build_transformers_generation_kwargs(
            self.backend.generation_kwargs,
            params,
        )
        try:
            text = self.backend.complete(prompt, generation_kwargs=generation_kwargs)
        except Exception as exc:
            raise _map_backend_exception(exc) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(
            model=model,
            text=text,
            provider=self.provider_name,
            finish_reason="completed",
            latency_ms=latency_ms,
            raw_output={
                "backend": self.provider_name,
                "requested_model": model,
                "response_schema": params.response_schema,
                "provider_options": dict(params.provider_options),
            },
        )

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Stream Transformers output as normalized events."""
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)


def build_backend_adapter(
    backend: BackendName,
    *,
    openai_config: OpenAIBackendConfig,
    llama_backend: LlamaCppServerBackend | None,
    transformers_backend: TransformersBackend | None,
) -> LLMProviderAdapter:
    """Build an adapter instance for one validated backend identifier.

    Centralizing this mapping ensures all client call paths resolve backends in
    the same deterministic way.

    Args:
        backend: Backend identifier to resolve.
        openai_config: OpenAI backend configuration payload.
        llama_backend: Optional llama-cpp-server backend instance.
        transformers_backend: Optional Transformers backend instance.

    Returns:
        Provider adapter instance for the requested backend.

    Raises:
        LLMInvalidRequestError: If the backend identifier is unsupported.
    """
    # Keep backend routing centralized so every call path uses the same mapping.
    if backend == "echo-test":
        return EchoTestProviderAdapter()
    if backend == "openai":
        return OpenAIProviderAdapter(config=openai_config)
    if backend == "llama-cpp-server":
        return LlamaCppServerProviderAdapter(backend=llama_backend)
    if backend == "transformers":
        return TransformersProviderAdapter(backend=transformers_backend)
    raise LLMInvalidRequestError(f"Unsupported backend '{backend}'.")


def _messages_to_prompt(
    messages: Sequence[LLMMessage],
    *,
    response_schema: dict[str, object] | None,
) -> str:
    """Combine ordered chat messages into one plain prompt string.

    The helper preserves message order and appends compact schema instructions
    when a response schema is supplied.

    Args:
        messages: Ordered chat messages to include in the prompt.
        response_schema: Optional response schema for structured output hints.

    Returns:
        Combined prompt text with optional schema guidance.
    """
    # Preserve order exactly; providers rely on message chronology.
    segments = [f"{message.role}: {message.content}" for message in messages]
    prompt = "\n".join(segments)
    schema_instruction = _build_schema_instruction(response_schema)
    if schema_instruction is None:
        return prompt

    return f"{prompt}\n\n{schema_instruction}"


def _build_transformers_generation_kwargs(
    defaults: Mapping[str, object],
    params: LLMChatParams,
) -> dict[str, object]:
    generation_kwargs = dict(defaults)
    generation_kwargs.update(_extract_generation_kwargs(params.provider_options))

    if params.temperature is not None:
        generation_kwargs["temperature"] = params.temperature
        if params.temperature > 0:
            generation_kwargs.setdefault("do_sample", True)
    if params.max_tokens is not None:
        generation_kwargs["max_new_tokens"] = params.max_tokens

    return generation_kwargs


def _extract_generation_kwargs(provider_options: Mapping[str, object]) -> dict[str, object]:
    raw = provider_options.get("generation_kwargs")
    if not isinstance(raw, Mapping):
        return {}
    return {key: value for key, value in raw.items() if isinstance(key, str)}


def _build_schema_instruction(response_schema: dict[str, object] | None) -> str | None:
    """Build compact schema guidance appended to provider prompts.

    The generated instruction intentionally avoids embedding full JSON schema
    text to reduce prompt echoing and token overhead.

    Args:
        response_schema: Optional response schema mapping.

    Returns:
        Schema instruction string or ``None`` when no schema is provided.
    """
    if response_schema is None:
        return None

    instruction_parts = [
        "Return only valid JSON as your entire response.",
        "Do not repeat the prompt, instructions, or schema.",
    ]

    required_fields = _extract_schema_string_list(response_schema.get("required"))
    if required_fields:
        required_preview = ", ".join(required_fields[:12])
        instruction_parts.append(f"Required top-level keys: {required_preview}.")

    property_fields = _extract_schema_property_fields(response_schema.get("properties"))
    if property_fields:
        property_preview = "; ".join(property_fields[:12])
        instruction_parts.append(f"Field expectations: {property_preview}.")

    return " ".join(instruction_parts)


def _extract_schema_string_list(raw_value: object) -> list[str]:
    """Extract normalized list of non-empty strings from schema-like values.

    Non-list and non-string values are ignored.

    Args:
        raw_value: Raw schema value to normalize.

    Returns:
        List of normalized non-empty strings.
    """
    if not isinstance(raw_value, list):
        return []
    values: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized:
            values.append(normalized)
    return values


def _extract_schema_property_fields(raw_properties: object) -> list[str]:
    """Extract top-level field/type hints from schema ``properties`` mappings.

    Returned values are formatted for compact prompt instructions.

    Args:
        raw_properties: Raw schema properties mapping.

    Returns:
        List of formatted field/type hints.
    """
    if not isinstance(raw_properties, dict):
        return []

    fields: list[str] = []
    for key, schema_value in raw_properties.items():
        if not isinstance(key, str) or not isinstance(schema_value, dict):
            continue
        type_hint = _format_schema_type(schema_value.get("type"))
        if type_hint:
            fields.append(f"{key} ({type_hint})")
            continue
        fields.append(key)
    return fields


def _format_schema_type(raw_type: object) -> str:
    """Format schema ``type`` values into compact, human-readable text.

    Supports both string and list-of-string schema type forms.

    Args:
        raw_type: Raw schema ``type`` value.

    Returns:
        Normalized type string for prompt inclusion.
    """
    if isinstance(raw_type, str):
        normalized = raw_type.strip()
        return normalized

    if not isinstance(raw_type, list):
        return ""

    variants: list[str] = []
    for item in raw_type:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized:
            variants.append(normalized)
    return " | ".join(variants)


def _map_backend_exception(exc: Exception) -> LLMError:
    """Map backend/provider exceptions into normalized contract exceptions.

    Mapping prefers explicit status-code signals when present and falls back to
    message heuristics for local or compatibility backends.

    Args:
        exc: Exception raised by the backend adapter.

    Returns:
        Normalized contract exception.
    """
    # Avoid double-wrapping when callers already raised typed contract errors.
    if isinstance(exc, LLMError):
        return exc

    message = str(exc)
    lower_message = message.lower()
    status_code = getattr(exc, "status_code", None)

    # Prefer explicit provider status codes when available.
    if status_code in {401, 403}:
        return LLMAuthError(message)
    if status_code == 429:
        return LLMRateLimitError(message)
    if status_code in {400, 404, 422}:
        return LLMInvalidRequestError(message)

    if "api key" in lower_message or "not set" in lower_message:
        return LLMAuthError(message)
    if "rate limit" in lower_message:
        return LLMRateLimitError(message)
    if isinstance(exc, ValueError):
        return LLMInvalidRequestError(message)
    # Fall back to provider-generic error for unknown exception families.
    return LLMProviderError(message)
