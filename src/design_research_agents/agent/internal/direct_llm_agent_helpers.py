"""Shared helper utilities for direct LLM agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from design_research_agents.agent.internal.input_parsing import extract_prompt
from design_research_agents.agent.internal.prompt_alternatives import (
    append_alternatives_block,
    format_raw_alternatives,
    resolve_alternatives_prompt_target,
)
from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)


def generate_response(llm_client: LLMClient, llm_request: LLMRequest) -> LLMResponse:
    """Issue one non-streaming LLM request.

    Args:
        llm_client: Parameter value.
        llm_request: Parameter value.

    Returns:
        The resulting value.

    Raises:
        AttributeError: Raised when the client exposes neither ``generate`` nor ``chat``.
    """
    generate_fn = cast(
        Callable[[LLMRequest], LLMResponse] | None,
        getattr(llm_client, "generate", None),
    )
    if generate_fn is not None:
        return generate_fn(llm_request)

    chat_fn = cast(Any, getattr(llm_client, "chat", None))
    if chat_fn is not None:
        model = llm_request.model or llm_client.default_model()
        return cast(
            LLMResponse,
            chat_fn(
                list(llm_request.messages),
                model=model,
                params=_chat_params_from_request(llm_request),
            ),
        )
    raise AttributeError("LLM client does not expose generate() or compatible chat().")


def _chat_params_from_request(llm_request: LLMRequest) -> LLMChatParams:
    """Build chat params from a normalized ``LLMRequest`` payload.

    Args:
        llm_request: Normalized LLM request payload.

    Returns:
        Chat params passed to ``LLMClient.chat``.
    """
    return LLMChatParams(
        temperature=llm_request.temperature,
        max_tokens=llm_request.max_tokens,
        response_schema=llm_request.response_schema,
        provider_options=dict(llm_request.provider_options),
    )


def build_success_result(
    *,
    llm_response: LLMResponse,
    request_id: str,
    dependencies: Mapping[str, object],
    message_source: str,
    message_count: int,
    llm_request: LLMRequest,
) -> ExecutionResult:
    """Build a success result payload from one completed model response.

    Args:
        llm_response: Parameter value.
        request_id: Parameter value.
        dependencies: Parameter value.
        message_source: Parameter value.
        message_count: Parameter value.
        llm_request: Parameter value.

    Returns:
        The resulting value.
    """
    output: dict[str, object] = {
        "model": llm_response.model,
        "model_text": llm_response.text,
    }
    return ExecutionResult(
        output=output,
        success=True,
        tool_results=[],
        model_response=llm_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            "llm_call": {
                "source": "direct",
                "message_source": message_source,
                "message_count": message_count,
                "temperature": llm_request.temperature,
                "max_tokens": llm_request.max_tokens,
                "response_schema_supplied": llm_request.response_schema is not None,
                "provider_options_keys": sorted(llm_request.provider_options.keys()),
            },
        },
    )


def extract_messages(
    *,
    input_payload: Mapping[str, object],
    default_system_prompt: str | None,
) -> tuple[list[LLMMessage], str]:
    """Extract normalized message list from explicit messages or prompt fallback.

    Args:
        input_payload: Parameter value.
        default_system_prompt: Parameter value.

    Returns:
        The resulting value.
    """
    normalized_input_messages = normalize_messages(input_payload.get("messages"))
    if normalized_input_messages:
        return (
            inject_alternatives_into_messages(
                messages=normalized_input_messages,
                input_payload=input_payload,
            ),
            "messages",
        )

    messages: list[LLMMessage] = []
    system_prompt = extract_system_prompt(
        input_payload=input_payload,
        default_system_prompt=default_system_prompt,
    )
    if system_prompt is not None:
        messages.append(LLMMessage(role="system", content=system_prompt))
    messages.append(LLMMessage(role="user", content=extract_prompt(input_payload)))
    return (
        inject_alternatives_into_messages(messages=messages, input_payload=input_payload),
        "prompt",
    )


def extract_system_prompt(
    *,
    input_payload: Mapping[str, object],
    default_system_prompt: str | None,
) -> str | None:
    """Extract optional system prompt override from run input or defaults.

    Args:
        input_payload: Parameter value.
        default_system_prompt: Parameter value.

    Returns:
        The resulting value.
    """
    raw_system_prompt = input_payload.get("system_prompt", default_system_prompt)
    if raw_system_prompt is None:
        return None
    normalized_system_prompt = str(raw_system_prompt).strip()
    return normalized_system_prompt or None


def normalize_messages(raw_messages: object) -> list[LLMMessage]:
    """Normalize optional message payload into a validated ``LLMMessage`` list.

    Args:
        raw_messages: Parameter value.

    Returns:
        The resulting value.
    """
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        return []

    normalized_messages: list[LLMMessage] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            continue
        raw_role = raw_message.get("role")
        if raw_role not in {"system", "user", "assistant", "tool"}:
            continue
        raw_content = raw_message.get("content")
        if not isinstance(raw_content, str):
            continue
        raw_name = raw_message.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
        normalized_messages.append(
            LLMMessage(role=raw_role, content=raw_content, name=name),
        )
    return normalized_messages


def inject_alternatives_into_messages(
    *,
    messages: Sequence[LLMMessage],
    input_payload: Mapping[str, object],
) -> list[LLMMessage]:
    """Inject optional alternatives context into either system or user prompt.

    Args:
        messages: Parameter value.
        input_payload: Parameter value.

    Returns:
        The resulting value.
    """
    alternatives_text = format_raw_alternatives(input_payload.get("alternatives"))
    if not alternatives_text:
        return list(messages)

    target = resolve_alternatives_prompt_target(input_payload=input_payload)
    if target == "system":
        return inject_alternatives_into_system_message(
            messages=messages, alternatives_text=alternatives_text
        )
    return inject_alternatives_into_user_message(
        messages=messages, alternatives_text=alternatives_text
    )


def inject_alternatives_into_system_message(
    *,
    messages: Sequence[LLMMessage],
    alternatives_text: str,
) -> list[LLMMessage]:
    """Inject alternatives text into the first system message or create one.

    Args:
        messages: Parameter value.
        alternatives_text: Parameter value.

    Returns:
        The resulting value.
    """
    injected_messages = list(messages)
    for index, message in enumerate(injected_messages):
        if message.role != "system":
            continue
        injected_messages[index] = LLMMessage(
            role="system",
            content=append_alternatives_block(
                prompt_text=message.content,
                section_label="Available alternatives",
                alternatives_text=alternatives_text,
            ),
            name=message.name,
        )
        return injected_messages

    return [
        LLMMessage(
            role="system",
            content=append_alternatives_block(
                prompt_text="Use these alternatives when producing your response.",
                section_label="Available alternatives",
                alternatives_text=alternatives_text,
            ),
        ),
        *injected_messages,
    ]


def inject_alternatives_into_user_message(
    *,
    messages: Sequence[LLMMessage],
    alternatives_text: str,
) -> list[LLMMessage]:
    """Inject alternatives text into the last user message or append one.

    Args:
        messages: Parameter value.
        alternatives_text: Parameter value.

    Returns:
        The resulting value.
    """
    injected_messages = list(messages)
    for index in range(len(injected_messages) - 1, -1, -1):
        message = injected_messages[index]
        if message.role != "user":
            continue
        injected_messages[index] = LLMMessage(
            role="user",
            content=append_alternatives_block(
                prompt_text=message.content,
                section_label="Available alternatives",
                alternatives_text=alternatives_text,
            ),
            name=message.name,
        )
        return injected_messages

    injected_messages.append(
        LLMMessage(
            role="user",
            content=append_alternatives_block(
                prompt_text="Choose from the alternatives below when relevant.",
                section_label="Available alternatives",
                alternatives_text=alternatives_text,
            ),
        )
    )
    return injected_messages


def extract_response_schema(
    input_payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Extract optional response-schema mapping from run input.

    Args:
        input_payload: Parameter value.

    Returns:
        The resulting value.
    """
    raw_response_schema = input_payload.get("response_schema")
    if isinstance(raw_response_schema, Mapping):
        return {key: value for key, value in raw_response_schema.items() if isinstance(key, str)}
    return None


def extract_temperature(
    *,
    input_payload: Mapping[str, object],
    default_value: float | None,
) -> float | None:
    """Extract optional sampling temperature from input payload.

    Args:
        input_payload: Parameter value.
        default_value: Parameter value.

    Returns:
        The resulting value.
    """
    raw_temperature = input_payload.get("temperature", default_value)
    if isinstance(raw_temperature, (int, float)):
        return float(raw_temperature)
    if isinstance(raw_temperature, str):
        normalized = raw_temperature.strip()
        if not normalized:
            return default_value
        try:
            return float(normalized)
        except ValueError:
            return default_value
    return default_value


def extract_max_tokens(
    *,
    input_payload: Mapping[str, object],
    default_value: int | None,
) -> int | None:
    """Extract optional positive max-token value from input payload.

    Args:
        input_payload: Parameter value.
        default_value: Parameter value.

    Returns:
        The resulting value.
    """
    raw_max_tokens = input_payload.get("max_tokens", default_value)
    if isinstance(raw_max_tokens, int):
        return raw_max_tokens if raw_max_tokens > 0 else default_value
    if isinstance(raw_max_tokens, str):
        normalized = raw_max_tokens.strip()
        if normalized.isdigit():
            parsed = int(normalized)
            return parsed if parsed > 0 else default_value
    return default_value


def merge_provider_options(
    *,
    default_provider_options: Mapping[str, object],
    raw_provider_options: object,
) -> dict[str, object]:
    """Merge default and input provider options into a plain dictionary.

    Args:
        default_provider_options: Parameter value.
        raw_provider_options: Parameter value.

    Returns:
        The resulting value.
    """
    merged = dict(default_provider_options)
    merged.update(coerce_provider_options(raw_provider_options))
    return merged


def coerce_provider_options(raw_provider_options: object) -> dict[str, object]:
    """Normalize optional provider options into ``dict[str, object]``.

    Args:
        raw_provider_options: Parameter value.

    Returns:
        The resulting value.
    """
    if not isinstance(raw_provider_options, Mapping):
        return {}
    return {key: value for key, value in raw_provider_options.items() if isinstance(key, str)}


__all__ = [
    "build_success_result",
    "coerce_provider_options",
    "extract_max_tokens",
    "extract_messages",
    "extract_response_schema",
    "extract_temperature",
    "generate_response",
    "merge_provider_options",
]
