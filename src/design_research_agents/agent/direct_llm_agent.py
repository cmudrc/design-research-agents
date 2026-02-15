"""Direct LLM agent that forwards requests without any tool orchestration.

The agent builds a minimal chat payload, calls the configured ``LLMClient``,
and returns the response as a standard ``AgentResult``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent._model_resolution import resolve_agent_model
from design_research_agents.agent._prompt_alternatives import (
    append_alternatives_block,
    format_raw_alternatives,
    resolve_alternatives_prompt_target,
)
from design_research_agents.agent._run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)


class DirectLLMAgent(Agent):
    """Agent that performs one direct model call with no tool runtime."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model: str | None = None,
        default_system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider_options: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize a direct-LLM agent with optional default generation args.

        Args:
            llm_client: LLM client used for prompt execution.
            model: Optional model override applied to all runs when provided.
            default_system_prompt: Optional default system prompt.
            temperature: Optional default sampling temperature.
            max_tokens: Optional default output-token cap.
            provider_options: Optional default backend-specific options.
        """
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when provided.")

        self._llm_client = llm_client
        self._model = model
        self._default_system_prompt = default_system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._provider_options = (
            _coerce_provider_options(provider_options) if provider_options is not None else {}
        )

    def run(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run one direct model call and return normalized ``AgentResult`` output."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(input)
        resolved_model, messages, message_source, llm_params = self._prepare_request(
            normalized_input
        )
        llm_response = self._llm_client.chat(messages, model=resolved_model, params=llm_params)
        return _build_success_result(
            llm_response=llm_response,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            message_source=message_source,
            message_count=len(messages),
            llm_params=llm_params,
        )

    def run_stream(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Stream direct model output and emit a final completion event."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(input)
        resolved_model, messages, message_source, llm_params = self._prepare_request(
            normalized_input
        )
        stream = self._llm_client.stream_chat(messages, model=resolved_model, params=llm_params)

        delta_parts: list[str] = []
        completed_response: LLMResponse | None = None
        for stream_event in stream:
            if stream_event.kind == "delta":
                delta_text = stream_event.delta_text or ""
                if delta_text:
                    delta_parts.append(delta_text)
                yield AgentStreamEvent(kind="delta", delta_text=delta_text)
                continue
            if stream_event.kind == "completed":
                completed_response = stream_event.response

        if completed_response is None:
            completed_response = LLMResponse(
                model=resolved_model,
                text="".join(delta_parts),
                provider=None,
            )
        elif not completed_response.text and delta_parts:
            completed_response = LLMResponse(
                model=completed_response.model,
                text="".join(delta_parts),
                provider=completed_response.provider,
                finish_reason=completed_response.finish_reason,
                usage=completed_response.usage,
                latency_ms=completed_response.latency_ms,
                raw_output=completed_response.raw_output,
            )

        if completed_response is not None:
            result = _build_success_result(
                llm_response=completed_response,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                message_source=message_source,
                message_count=len(messages),
                llm_params=llm_params,
            )
            yield AgentStreamEvent(kind="completed", result=result)
        else:
            yield AgentStreamEvent(kind="failed", result=None)

    def _prepare_request(
        self,
        input_payload: Mapping[str, object],
    ) -> tuple[str, list[LLMMessage], str, LLMChatParams]:
        """Resolve model/messages/params into one reusable request payload."""
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
            input_payload=input_payload,
            init_model=self._model,
        )
        messages, message_source = _extract_messages(
            input_payload=input_payload,
            default_system_prompt=self._default_system_prompt,
        )
        llm_params = LLMChatParams(
            temperature=_extract_temperature(
                input_payload=input_payload,
                default_value=self._temperature,
            ),
            max_tokens=_extract_max_tokens(
                input_payload=input_payload,
                default_value=self._max_tokens,
            ),
            response_schema=_extract_response_schema(input_payload),
            provider_options=_merge_provider_options(
                default_provider_options=self._provider_options,
                raw_provider_options=input_payload.get("provider_options"),
            ),
        )
        return resolved_model, messages, message_source, llm_params


def _build_success_result(
    *,
    llm_response: LLMResponse,
    request_id: str,
    dependencies: Mapping[str, object],
    message_source: str,
    message_count: int,
    llm_params: LLMChatParams,
) -> AgentResult:
    """Build a success result payload from one completed model response."""
    output: dict[str, object] = {
        "model": llm_response.model,
        "model_text": llm_response.text,
    }
    return AgentResult(
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
                "temperature": llm_params.temperature,
                "max_tokens": llm_params.max_tokens,
                "response_schema_supplied": llm_params.response_schema is not None,
                "provider_options_keys": sorted(llm_params.provider_options.keys()),
            },
        },
    )


def _extract_messages(
    *,
    input_payload: Mapping[str, object],
    default_system_prompt: str | None,
) -> tuple[list[LLMMessage], str]:
    """Extract normalized message list from explicit messages or prompt fallback."""
    normalized_input_messages = _normalize_messages(input_payload.get("messages"))
    if normalized_input_messages:
        return (
            _inject_alternatives_into_messages(
                messages=normalized_input_messages,
                input_payload=input_payload,
            ),
            "messages",
        )

    messages: list[LLMMessage] = []
    system_prompt = _extract_system_prompt(
        input_payload=input_payload,
        default_system_prompt=default_system_prompt,
    )
    if system_prompt is not None:
        messages.append(LLMMessage(role="system", content=system_prompt))
    messages.append(LLMMessage(role="user", content=_extract_prompt(input_payload)))
    return (
        _inject_alternatives_into_messages(messages=messages, input_payload=input_payload),
        "prompt",
    )


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from input payload with stable fallback text."""
    raw_prompt = input_payload.get(
        "prompt",
        input_payload.get("text", "Provide a concise response."),
    )
    return str(raw_prompt)


def _extract_system_prompt(
    *,
    input_payload: Mapping[str, object],
    default_system_prompt: str | None,
) -> str | None:
    """Extract optional system prompt override from run input or defaults."""
    raw_system_prompt = input_payload.get("system_prompt", default_system_prompt)
    if raw_system_prompt is None:
        return None
    normalized_system_prompt = str(raw_system_prompt).strip()
    return normalized_system_prompt or None


def _normalize_messages(raw_messages: object) -> list[LLMMessage]:
    """Normalize optional message payload into a validated ``LLMMessage`` list."""
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


def _inject_alternatives_into_messages(
    *,
    messages: Sequence[LLMMessage],
    input_payload: Mapping[str, object],
) -> list[LLMMessage]:
    """Inject optional alternatives context into either system or user prompt."""
    alternatives_text = format_raw_alternatives(input_payload.get("alternatives"))
    if not alternatives_text:
        return list(messages)

    target = resolve_alternatives_prompt_target(input_payload=input_payload)
    if target == "system":
        return _inject_alternatives_into_system_message(
            messages=messages, alternatives_text=alternatives_text
        )
    return _inject_alternatives_into_user_message(
        messages=messages, alternatives_text=alternatives_text
    )


def _inject_alternatives_into_system_message(
    *,
    messages: Sequence[LLMMessage],
    alternatives_text: str,
) -> list[LLMMessage]:
    """Inject alternatives text into the first system message or create one."""
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


def _inject_alternatives_into_user_message(
    *,
    messages: Sequence[LLMMessage],
    alternatives_text: str,
) -> list[LLMMessage]:
    """Inject alternatives text into the last user message or append one."""
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


def _extract_response_schema(
    input_payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Extract optional response-schema mapping from run input."""
    raw_response_schema = input_payload.get("response_schema")
    if isinstance(raw_response_schema, Mapping):
        return {key: value for key, value in raw_response_schema.items() if isinstance(key, str)}
    return None


def _extract_temperature(
    *,
    input_payload: Mapping[str, object],
    default_value: float | None,
) -> float | None:
    """Extract optional sampling temperature from input payload."""
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


def _extract_max_tokens(
    *,
    input_payload: Mapping[str, object],
    default_value: int | None,
) -> int | None:
    """Extract optional positive max-token value from input payload."""
    raw_max_tokens = input_payload.get("max_tokens", default_value)
    if isinstance(raw_max_tokens, int):
        return raw_max_tokens if raw_max_tokens > 0 else default_value
    if isinstance(raw_max_tokens, str):
        normalized = raw_max_tokens.strip()
        if normalized.isdigit():
            parsed = int(normalized)
            return parsed if parsed > 0 else default_value
    return default_value


def _merge_provider_options(
    *,
    default_provider_options: Mapping[str, object],
    raw_provider_options: object,
) -> dict[str, object]:
    """Merge default and input provider options into a plain dictionary."""
    merged = dict(default_provider_options)
    merged.update(_coerce_provider_options(raw_provider_options))
    return merged


def _coerce_provider_options(raw_provider_options: object) -> dict[str, object]:
    """Normalize optional provider options into ``dict[str, object]``."""
    if not isinstance(raw_provider_options, Mapping):
        return {}
    return {key: value for key, value in raw_provider_options.items() if isinstance(key, str)}
