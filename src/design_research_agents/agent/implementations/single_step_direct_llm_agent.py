"""Direct LLM agent that forwards requests without any tool orchestration.

The agent builds a minimal chat payload, calls the configured ``LLMClient``,
and returns the response as a standard ``AgentResult``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent.internal.input_parsing import extract_prompt as _extract_prompt
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.prompt_alternatives import (
    append_alternatives_block,
    format_raw_alternatives,
    resolve_alternatives_prompt_target,
)
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.agent.internal.streaming import (
    StreamAccumulator,
    finalize_stream_response,
)
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMClient,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.tracing import (
    Tracer,
    emit_model_token,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


class SingleStepDirectLLMAgent(Agent):
    """Agent that performs one direct model call with no tool runtime."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider_options: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a direct-LLM agent with optional default generation args.

        Args:
            llm_client: LLM client used for prompt execution.
            system_prompt: Optional default system prompt.
            temperature: Optional default sampling temperature.
            max_tokens: Optional default output-token cap.
            provider_options: Optional default backend-specific options.
            tracer: Optional explicit tracer dependency.
        """
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when provided.")

        self._llm_client = llm_client
        self._default_system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tracer = tracer
        self._provider_options = (
            _coerce_provider_options(provider_options) if provider_options is not None else {}
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run one direct model call and return normalized ``AgentResult`` output.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        resolved_model, messages, message_source, llm_request = self._prepare_request(
            normalized_input,
            request_id=resolved_request_id,
        )
        trace_scope = start_trace_run(
            agent_name="SingleStepDirectLLMAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=messages,
            params=llm_request,
            metadata={"agent": "SingleStepDirectLLMAgent", "message_source": message_source},
        )
        try:
            llm_response = _generate_response(self._llm_client, llm_request)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            finish_trace_run(trace_scope, error=str(exc))
            raise

        finish_model_call(model_span_id, response=llm_response)
        result = _build_success_result(
            llm_response=llm_response,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            message_source=message_source,
            message_count=len(messages),
            llm_request=llm_request,
        )
        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Stream direct model output and emit a final completion event.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Yields:
            Streaming events through completion.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        resolved_model, messages, message_source, llm_request = self._prepare_request(
            normalized_input,
            request_id=resolved_request_id,
        )
        trace_scope = start_trace_run(
            agent_name="SingleStepDirectLLMAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=messages,
            params=llm_request,
            metadata={"agent": "SingleStepDirectLLMAgent", "message_source": message_source},
        )
        try:
            stream = _stream_response(self._llm_client, llm_request)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            finish_trace_run(trace_scope, error=str(exc))
            raise

        accumulator = StreamAccumulator()
        try:
            for stream_event in stream:
                accumulator.apply(stream_event)
                delta_text = stream_event.text_delta or ""
                if delta_text:
                    emit_model_token(model_span_id, delta_text=delta_text)
                yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            finish_trace_run(trace_scope, error=str(exc))
            raise

        completed_response = finalize_stream_response(
            stream=stream,
            accumulator=accumulator,
            model=resolved_model,
        )
        finish_model_call(model_span_id, response=completed_response)
        result = _build_success_result(
            llm_response=completed_response,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            message_source=message_source,
            message_count=len(messages),
            llm_request=llm_request,
        )
        finish_trace_run(trace_scope, result=result)
        yield AgentStreamEvent(kind="completed", result=result)

    def _prepare_request(
        self,
        input_payload: Mapping[str, object],
        *,
        request_id: str | None,
    ) -> tuple[str, list[LLMMessage], str, LLMRequest]:
        """Resolve model/messages/params into one reusable request payload.

        Args:
            input_payload: Normalized run input payload mapping.
            request_id: Optional request identifier used in metadata.

        Returns:
            Tuple of resolved model, messages, message source, and chat params.
        """
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        messages, message_source = _extract_messages(
            input_payload=input_payload,
            default_system_prompt=self._default_system_prompt,
        )
        llm_request = LLMRequest(
            messages=messages,
            model=resolved_model,
            temperature=_extract_temperature(
                input_payload=input_payload,
                default_value=self._temperature,
            ),
            max_tokens=_extract_max_tokens(
                input_payload=input_payload,
                default_value=self._max_tokens,
            ),
            response_schema=_extract_response_schema(input_payload),
            metadata={
                "request_id": request_id,
                "agent": "SingleStepDirectLLMAgent",
                "message_source": message_source,
            },
            provider_options=_merge_provider_options(
                default_provider_options=self._provider_options,
                raw_provider_options=input_payload.get("provider_options"),
            ),
        )
        return resolved_model, messages, message_source, llm_request


def _generate_response(llm_client: LLMClient, llm_request: LLMRequest) -> LLMResponse:
    return llm_client.generate(llm_request)


def _stream_response(llm_client: LLMClient, llm_request: LLMRequest) -> Iterator[LLMDelta]:
    return llm_client.stream(llm_request)


def _build_success_result(
    *,
    llm_response: LLMResponse,
    request_id: str,
    dependencies: Mapping[str, object],
    message_source: str,
    message_count: int,
    llm_request: LLMRequest,
) -> AgentResult:
    """Build a success result payload from one completed model response.

    Args:
        llm_response: Completed LLM response payload.
        request_id: Request identifier for tracing metadata.
        dependencies: Dependency payload mapping for the run.
        message_source: Message source label (e.g. ``prompt`` or ``messages``).
        message_count: Number of messages sent to the model.
        llm_request: Request payload sent to the LLM backend.

    Returns:
        Agent result payload describing the successful run.
    """
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
                "temperature": llm_request.temperature,
                "max_tokens": llm_request.max_tokens,
                "response_schema_supplied": llm_request.response_schema is not None,
                "provider_options_keys": sorted(llm_request.provider_options.keys()),
            },
        },
    )


def _extract_messages(
    *,
    input_payload: Mapping[str, object],
    default_system_prompt: str | None,
) -> tuple[list[LLMMessage], str]:
    """Extract normalized message list from explicit messages or prompt fallback.

    Args:
        input_payload: Normalized run input payload mapping.
        default_system_prompt: Optional default system prompt override.

    Returns:
        Tuple of normalized messages and the source label.
    """
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


def _extract_system_prompt(
    *,
    input_payload: Mapping[str, object],
    default_system_prompt: str | None,
) -> str | None:
    """Extract optional system prompt override from run input or defaults.

    Args:
        input_payload: Normalized run input payload mapping.
        default_system_prompt: Optional default system prompt override.

    Returns:
        System prompt text when provided, otherwise ``None``.
    """
    raw_system_prompt = input_payload.get("system_prompt", default_system_prompt)
    if raw_system_prompt is None:
        return None
    normalized_system_prompt = str(raw_system_prompt).strip()
    return normalized_system_prompt or None


def _normalize_messages(raw_messages: object) -> list[LLMMessage]:
    """Normalize optional message payload into a validated ``LLMMessage`` list.

    Args:
        raw_messages: Raw messages payload from input.

    Returns:
        Normalized list of LLM messages.
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


def _inject_alternatives_into_messages(
    *,
    messages: Sequence[LLMMessage],
    input_payload: Mapping[str, object],
) -> list[LLMMessage]:
    """Inject optional alternatives context into either system or user prompt.

    Args:
        messages: Existing message sequence.
        input_payload: Normalized run input payload mapping.

    Returns:
        Message list with alternatives injected when available.
    """
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
    """Inject alternatives text into the first system message or create one.

    Args:
        messages: Existing message sequence.
        alternatives_text: Alternatives block text.

    Returns:
        Message list with alternatives injected into the system prompt.
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


def _inject_alternatives_into_user_message(
    *,
    messages: Sequence[LLMMessage],
    alternatives_text: str,
) -> list[LLMMessage]:
    """Inject alternatives text into the last user message or append one.

    Args:
        messages: Existing message sequence.
        alternatives_text: Alternatives block text.

    Returns:
        Message list with alternatives injected into the user prompt.
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


def _extract_response_schema(
    input_payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Extract optional response-schema mapping from run input.

    Args:
        input_payload: Normalized run input payload mapping.

    Returns:
        Response schema mapping when present, otherwise ``None``.
    """
    raw_response_schema = input_payload.get("response_schema")
    if isinstance(raw_response_schema, Mapping):
        return {key: value for key, value in raw_response_schema.items() if isinstance(key, str)}
    return None


def _extract_temperature(
    *,
    input_payload: Mapping[str, object],
    default_value: float | None,
) -> float | None:
    """Extract optional sampling temperature from input payload.

    Args:
        input_payload: Normalized run input payload mapping.
        default_value: Default temperature when no valid override is provided.

    Returns:
        Parsed temperature value or default.
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


def _extract_max_tokens(
    *,
    input_payload: Mapping[str, object],
    default_value: int | None,
) -> int | None:
    """Extract optional positive max-token value from input payload.

    Args:
        input_payload: Normalized run input payload mapping.
        default_value: Default max tokens when no valid override is provided.

    Returns:
        Parsed max tokens value or default.
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


def _merge_provider_options(
    *,
    default_provider_options: Mapping[str, object],
    raw_provider_options: object,
) -> dict[str, object]:
    """Merge default and input provider options into a plain dictionary.

    Args:
        default_provider_options: Default provider options mapping.
        raw_provider_options: Raw provider options payload to merge.

    Returns:
        Merged provider options dictionary.
    """
    merged = dict(default_provider_options)
    merged.update(_coerce_provider_options(raw_provider_options))
    return merged


def _coerce_provider_options(raw_provider_options: object) -> dict[str, object]:
    """Normalize optional provider options into ``dict[str, object]``.

    Args:
        raw_provider_options: Raw provider options payload.

    Returns:
        Normalized provider options dictionary.
    """
    if not isinstance(raw_provider_options, Mapping):
        return {}
    return {key: value for key, value in raw_provider_options.items() if isinstance(key, str)}
