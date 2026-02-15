"""Tool-calling agent that chooses a tool and arguments from model output.

The agent prompts the model with runtime-backed tool options, validates the
structured response, and executes one tool call with deterministic fallbacks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.agent._model_resolution import resolve_agent_model
from design_research_agents.agent._prompt_alternatives import (
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    resolve_alternatives_prompt_target,
)
from design_research_agents.agent._response_schemas import (
    build_tool_call_response_schema,
    clone_response_schema,
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
    LLMRequest,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolRuntime, ToolSpec
from design_research_agents.prompts import load_prompt, render_prompt
from design_research_agents.tracing import (
    emit_tool_selection_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


@dataclass(slots=True, frozen=True)
class _ToolChoice:
    """Normalized tool option used by planning, validation, and fallback logic.

    Attributes:
        tool_name: Runtime tool identifier.
        description: Human-readable tool description shown to the model.
        input_schema: JSON-schema-like payload shape expected by the tool.
    """

    tool_name: str
    description: str
    input_schema: dict[str, object]


class ToolCallingAgent(Agent):
    """Agent that asks the model to select a tool and structured arguments.

    The execution path is: gather choices, request strict JSON tool call, parse
    and validate, then invoke exactly one selected tool.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        model: str | None = None,
        default_tool_name: str = "text_stats_tool",
    ) -> None:
        """Initialize a tool-calling agent with injected runtime dependencies.

        Args:
            llm_client: LLM client used for prompt execution.
            tool_runtime: Tool runtime used for tool invocation.
            model: Optional model override applied to all runs when provided.
            default_tool_name: Fallback tool used when no explicit choices are supplied.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._model = model
        self._default_tool_name = default_tool_name
        self._runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        self._compiled_tool_choices = _extract_tool_choices(
            tool_specs=self._runtime_specs,
            default_tool_name=self._default_tool_name,
        )
        self._default_tool_call_response_schema = _tool_call_response_schema(
            [choice.tool_name for choice in self._compiled_tool_choices]
        )

    def run(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run one tool-calling step from planning through tool execution.

        The run prompts for a structured tool call, validates selection, resolves
        tool input, executes the tool, and returns unified output/metadata.

        Args:
            input: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(input)
        trace_scope = start_trace_run(
            agent_name="ToolCallingAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
        )
        prompt = _extract_prompt(normalized_input)
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
            input_payload=normalized_input,
            init_model=self._model,
        )
        choices = [_clone_tool_choice(choice) for choice in self._compiled_tool_choices]
        alternatives_prompt_target = resolve_alternatives_prompt_target(
            input_payload=normalized_input
        )
        choices_text = _build_tool_choices_text(choices=choices)
        choices_block = build_user_prompt_alternatives_block(
            section_label="Available tools",
            alternatives_text=choices_text,
            target=alternatives_prompt_target,
        )
        user_prompt = _build_tool_call_prompt(prompt=prompt, choices_block=choices_block)
        system_prompt = load_prompt("tool_calling_system")
        if alternatives_prompt_target == "system":
            system_prompt = append_alternatives_block(
                prompt_text=system_prompt,
                section_label="Available tools",
                alternatives_text=choices_text,
            )

        model_messages = [
            LLMMessage(
                role="system",
                content=system_prompt,
            ),
            LLMMessage(
                role="user",
                content=user_prompt,
            ),
        ]
        llm_request = LLMRequest(
            messages=model_messages,
            model=resolved_model,
            tools=list(self._runtime_specs.values()),
            metadata={
                "request_id": resolved_request_id,
                "agent": "ToolCallingAgent",
            },
            provider_options={"agent": "ToolCallingAgent"},
        )
        model_call_payload: LLMRequest | LLMChatParams
        if _supports_generate(self._llm_client):
            model_call_payload = llm_request
        else:
            model_call_payload = LLMChatParams(
                response_schema=clone_response_schema(self._default_tool_call_response_schema),
                provider_options=dict(llm_request.provider_options),
            )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=model_messages,
            params=model_call_payload,
            metadata={"agent": "ToolCallingAgent"},
        )
        try:
            llm_response = _request_tool_call_response(
                llm_client=self._llm_client,
                llm_request=llm_request,
                response_schema=self._default_tool_call_response_schema,
            )
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            finish_trace_run(trace_scope, error=str(exc))
            raise
        finish_model_call(model_span_id, response=llm_response)

        parsed_tool_call = _parse_tool_call_from_response(llm_response)
        if parsed_tool_call is None:
            parsed_tool_call = _parse_tool_call(llm_response.text)
        selected_choice, tool_call_source, tool_call_reason = _select_tool_choice(
            parsed_tool_call=parsed_tool_call,
            prompt=prompt,
            choices=choices,
        )
        emit_tool_selection_decision(
            source=tool_call_source,
            tool_name=selected_choice.tool_name,
            reason=tool_call_reason,
            parsed_tool_call=parsed_tool_call,
        )
        tool_input = _resolve_tool_input(
            selected_choice=selected_choice,
            parsed_tool_call=parsed_tool_call,
            input_payload=normalized_input,
            llm_response_text=llm_response.text,
        )

        tool_result = self._tool_runtime.invoke(
            selected_choice.tool_name,
            tool_input,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
        )
        output: dict[str, object] = {
            "model_text": llm_response.text,
            "tool_name": selected_choice.tool_name,
            "tool_input": tool_input,
            "tool_output": tool_result.output,
        }
        result = AgentResult(
            output=output,
            success=tool_result.success,
            tool_results=[tool_result],
            model_response=llm_response,
            metadata={
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "tool_call": {
                    "source": tool_call_source,
                    "reason": tool_call_reason,
                    "available_tools": [choice.tool_name for choice in choices],
                    "parsed_tool_call": parsed_tool_call,
                },
            },
        )
        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Emit a deterministic stream wrapper around ``run``.

        The wrapper emits one delta containing full model text, followed by a
        completion event with the final ``AgentResult``.

        Args:
            input: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Yields:
            Streaming events through completion.
        """
        result = self.run(input, request_id=request_id, dependencies=dependencies)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from run input.

    Falls back to ``text`` and then a default string when missing.

    Args:
        input_payload: Normalized run input payload mapping.

    Returns:
        Prompt text for the run.
    """
    raw_prompt = input_payload.get(
        "prompt", input_payload.get("text", "Provide a concise response.")
    )
    return str(raw_prompt)


def _extract_tool_choices(
    *,
    tool_specs: Mapping[str, ToolSpec],
    default_tool_name: str,
) -> list[_ToolChoice]:
    """Extract normalized tool choices from runtime specs.

    Args:
        tool_specs: Mapping of tool specifications from the runtime.
        default_tool_name: Fallback tool name when no specs are available.

    Returns:
        List of normalized tool choices.
    """
    if tool_specs:
        return [
            _ToolChoice(
                tool_name=spec.name,
                description=spec.description,
                input_schema=dict(spec.input_schema),
            )
            for spec in tool_specs.values()
        ]

    # Keep legacy one-tool behavior as a last resort.
    return [
        _ToolChoice(
            tool_name=default_tool_name,
            description="Default fallback route.",
            input_schema={"type": "object"},
        )
    ]


def _build_tool_call_prompt(*, prompt: str, choices_block: str) -> str:
    """Build prompt asking model to select tool and structured arguments.

    The prompt receives pre-rendered choices text so callers can route
    alternatives either through the user prompt or the system prompt.

    Args:
        prompt: User prompt text.
        choices_block: Pre-rendered choices block text.

    Returns:
        Rendered tool-call prompt text.
    """
    return render_prompt(
        "tool_calling_user_select_tool",
        variables={
            "choices_block": choices_block,
            "user_prompt": prompt,
        },
    )


def _build_tool_choices_text(*, choices: Sequence[_ToolChoice]) -> str:
    """Build formatted runtime tool choices text.

    Args:
        choices: Sequence of tool choices to render.

    Returns:
        Formatted tool choices block text.
    """
    choice_lines: list[str] = []
    for choice in choices:
        choice_lines.append(
            "\n".join(
                [
                    f"- tool_name: {choice.tool_name}",
                    f"  description: {choice.description or '(none)'}",
                    f"  input_schema: {json.dumps(choice.input_schema, sort_keys=True)}",
                ]
            )
        )
    return "\n".join(choice_lines)


def _clone_tool_choice(choice: _ToolChoice) -> _ToolChoice:
    """Clone one tool choice so run-local payloads remain isolated.

    Args:
        choice: Tool choice to clone.

    Returns:
        Cloned tool choice instance.
    """
    return _ToolChoice(
        tool_name=choice.tool_name,
        description=choice.description,
        input_schema=dict(choice.input_schema),
    )


def _supports_generate(llm_client: LLMClient) -> bool:
    return callable(getattr(llm_client, "generate", None))


def _request_tool_call_response(
    *,
    llm_client: LLMClient,
    llm_request: LLMRequest,
    response_schema: Mapping[str, object],
) -> LLMResponse:
    generate_fn = getattr(llm_client, "generate", None)
    if callable(generate_fn):
        return generate_fn(llm_request)

    chat_fn = getattr(llm_client, "chat", None)
    if not callable(chat_fn):
        raise AttributeError("LLM client does not expose generate() or chat().")

    resolved_model = llm_request.model or _resolve_default_model(llm_client)
    llm_params = LLMChatParams(
        response_schema=clone_response_schema(dict(response_schema)),
        provider_options=dict(llm_request.provider_options),
    )
    return chat_fn(
        llm_request.messages,
        model=resolved_model,
        params=llm_params,
    )


def _resolve_default_model(llm_client: LLMClient) -> str:
    default_model_fn = getattr(llm_client, "default_model", None)
    if callable(default_model_fn):
        return str(default_model_fn())
    return "local-model"


def _tool_call_response_schema(available_tool_names: Sequence[str]) -> dict[str, object]:
    return build_tool_call_response_schema(
        tool_names=available_tool_names,
    )


def _parse_tool_call_from_response(llm_response: LLMResponse) -> dict[str, object] | None:
    if not llm_response.tool_calls:
        return None
    call = llm_response.tool_calls[0]
    try:
        tool_input = json.loads(call.arguments_json)
    except json.JSONDecodeError:
        tool_input = call.arguments_json
    return {
        "tool_name": call.name,
        "tool_input": tool_input,
        "call_id": call.call_id,
    }


def _parse_tool_call(raw_text: str) -> dict[str, object] | None:
    """Parse tool-call JSON payload from model text output.

    Supports strict JSON responses and JSON objects embedded in surrounding text.

    Args:
        raw_text: Raw model response text.

    Returns:
        Parsed tool-call mapping or ``None`` when parsing fails.
    """
    parsed = _load_json_mapping(raw_text)
    if parsed is not None:
        return parsed

    # Allow extra surrounding text by scanning for the first JSON object.
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _load_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Load text as a JSON mapping.

    Returns ``None`` when the text is invalid JSON or not an object.

    Args:
        raw_text: Raw text to parse as JSON.

    Returns:
        Parsed JSON mapping or ``None`` when invalid.
    """
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _select_tool_choice(
    *,
    parsed_tool_call: Mapping[str, object] | None,
    prompt: str,
    choices: Sequence[_ToolChoice],
) -> tuple[_ToolChoice, str, str]:
    """Select validated tool choice from model output or fallback routing.

    Model-provided choices are preferred when valid; otherwise deterministic
    lexical fallback scoring selects a tool.

    Args:
        parsed_tool_call: Parsed tool-call payload, if available.
        prompt: User prompt text.
        choices: Available tool choices.

    Returns:
        Tuple of selected choice, source label, and reason string.
    """
    allowed_names = {choice.tool_name for choice in choices}
    if parsed_tool_call is not None:
        raw_tool_name = parsed_tool_call.get("tool_name", parsed_tool_call.get("name"))
        if isinstance(raw_tool_name, str):
            selected_name = raw_tool_name.strip()
            if selected_name in allowed_names:
                selected_choice = next(
                    choice for choice in choices if choice.tool_name == selected_name
                )
                return selected_choice, "model", "validated model tool_name"

    fallback_choice, fallback_reason = _fallback_select_tool_choice(prompt=prompt, choices=choices)
    return fallback_choice, "fallback", fallback_reason


def _resolve_tool_input(
    *,
    selected_choice: _ToolChoice,
    parsed_tool_call: Mapping[str, object] | None,
    input_payload: Mapping[str, object],
    llm_response_text: str,
) -> dict[str, object]:
    """Resolve final tool input from model payload, run input, or heuristics.

    Args:
        selected_choice: Selected tool choice.
        parsed_tool_call: Parsed tool-call payload, if available.
        input_payload: Normalized run input payload mapping.
        llm_response_text: Raw model response text.

    Returns:
        Resolved tool input mapping.
    """
    if parsed_tool_call is not None:
        raw_tool_input = parsed_tool_call.get(
            "tool_input",
            parsed_tool_call.get("arguments", parsed_tool_call.get("args")),
        )
        normalized_from_model = _coerce_tool_input(raw_tool_input)
        if normalized_from_model:
            return normalized_from_model

    raw_tool_input = input_payload.get("tool_input")
    normalized_from_input = _coerce_tool_input(raw_tool_input)
    if normalized_from_input:
        return normalized_from_input

    if selected_choice.tool_name == "calculator_tool":
        expression = _infer_expression(
            input_payload=input_payload,
            prompt=_extract_prompt(input_payload),
        )
        return {"expression": expression}

    if selected_choice.tool_name == "text_stats_tool":
        analysis_text = input_payload.get("analysis_text")
        if analysis_text is not None:
            return {"text": str(analysis_text)}
        return {"text": llm_response_text}

    return {}


def _coerce_tool_input(raw_tool_input: object) -> dict[str, object] | None:
    """Convert raw tool-input payload into a JSON-like dictionary when possible.

    Supports direct mappings and JSON-encoded string payloads.

    Args:
        raw_tool_input: Raw tool input payload.

    Returns:
        Normalized tool input mapping, or ``None`` when invalid.
    """
    if isinstance(raw_tool_input, Mapping):
        return dict(raw_tool_input)
    if isinstance(raw_tool_input, str):
        parsed = _load_json_mapping(raw_tool_input)
        if parsed is not None:
            return parsed
    return None


def _fallback_select_tool_choice(
    *,
    prompt: str,
    choices: Sequence[_ToolChoice],
) -> tuple[_ToolChoice, str]:
    """Select fallback tool choice using deterministic lexical-signal scoring.

    Scoring combines token overlap and arithmetic/text intent signals.

    Args:
        prompt: User prompt text.
        choices: Available tool choices.

    Returns:
        Tuple of selected tool choice and fallback reason string.
    """
    prompt_text = prompt.lower()
    prompt_tokens = _tokenize(prompt_text)
    prompt_looks_math = _looks_like_arithmetic_request(prompt_text)
    prompt_looks_text = _looks_like_text_analysis_request(prompt_text)

    selected_choice = choices[0]
    selected_score = -1
    selected_reason = "fallback-first-choice"
    for choice in choices:
        searchable = " ".join([choice.tool_name, choice.description]).lower()
        route_tokens = _tokenize(searchable)
        score = len(prompt_tokens.intersection(route_tokens))
        reason_parts: list[str] = []
        if score > 0:
            reason_parts.append(f"token-overlap:{score}")
        if prompt_looks_math and _looks_like_arithmetic_tool(searchable):
            score += 5
            reason_parts.append("math-signal")
        if prompt_looks_text and _looks_like_text_tool(searchable):
            score += 5
            reason_parts.append("text-signal")

        if score > selected_score:
            selected_choice = choice
            selected_score = score
            selected_reason = ", ".join(reason_parts) if reason_parts else "fallback-first-choice"

    return selected_choice, selected_reason


def _infer_expression(*, input_payload: Mapping[str, object], prompt: str) -> str:
    """Infer calculator expression from payload fields and prompt text.

    Explicit expressions win; otherwise a regex candidate is extracted from the
    prompt before falling back to the full prompt text.

    Args:
        input_payload: Normalized run input payload mapping.
        prompt: User prompt text.

    Returns:
        Inferred arithmetic expression string.
    """
    explicit_expression = input_payload.get("expression")
    if explicit_expression is not None:
        return str(explicit_expression)

    text_expression = input_payload.get("text")
    if text_expression is not None:
        return str(text_expression)

    match = re.search(r"(\(?-?\d[\d\s\.\+\-\*\/%\(\)]*\d\)?)", prompt)
    if match is not None:
        expression = match.group(1).strip()
        if expression and any(operator in expression for operator in "+-*/%"):
            return expression

    return prompt


def _tokenize(text: str) -> set[str]:
    """Tokenize text into normalized alphanumeric words for lexical matching.

    Tokenization is intentionally simple and deterministic.

    Args:
        text: Text to tokenize.

    Returns:
        Set of normalized tokens.
    """
    return {token for token in re.findall(r"[a-z0-9_]+", text) if token}


def _looks_like_arithmetic_request(text: str) -> bool:
    """Return whether prompt text appears to request arithmetic computation.

    Uses regex patterns and keyword heuristics.

    Args:
        text: Prompt text to inspect.

    Returns:
        ``True`` when the text suggests arithmetic computation.
    """
    if re.search(r"\d+\s*[\+\-\*\/%]\s*\d+", text):
        return True
    math_keywords = {
        "calculate",
        "calculator",
        "arithmetic",
        "equation",
        "expression",
        "math",
        "solve",
        "sum",
        "multiply",
        "divide",
        "compute",
    }
    return any(keyword in text for keyword in math_keywords)


def _looks_like_text_analysis_request(text: str) -> bool:
    """Return whether prompt text appears to request text analysis.

    Uses lightweight keyword heuristics.

    Args:
        text: Prompt text to inspect.

    Returns:
        ``True`` when the text suggests text analysis.
    """
    text_keywords = {
        "text",
        "word",
        "words",
        "line",
        "lines",
        "character",
        "characters",
        "count",
        "stats",
        "statistics",
        "summary",
        "summarize",
    }
    return any(keyword in text for keyword in text_keywords)


def _looks_like_arithmetic_tool(text: str) -> bool:
    """Return whether tool description text appears arithmetic-focused.

    Uses tool-name/description keyword matching.

    Args:
        text: Tool name/description text to inspect.

    Returns:
        ``True`` when the text suggests an arithmetic tool.
    """
    arithmetic_tool_keywords = {
        "calc",
        "calculator",
        "math",
        "arithmetic",
        "expression",
        "equation",
        "compute",
    }
    return any(keyword in text for keyword in arithmetic_tool_keywords)


def _looks_like_text_tool(text: str) -> bool:
    """Return whether tool description text appears text-analysis-focused.

    Uses tool-name/description keyword matching.

    Args:
        text: Tool name/description text to inspect.

    Returns:
        ``True`` when the text suggests a text analysis tool.
    """
    text_tool_keywords = {
        "text",
        "word",
        "line",
        "character",
        "stats",
        "statistics",
        "summary",
        "summarize",
        "analy",
        "count",
    }
    return any(keyword in text for keyword in text_tool_keywords)
