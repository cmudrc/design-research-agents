"""Tool-calling agent that chooses a tool and arguments from model output."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMChatParams, LLMClient, LLMMessage
from design_research_agents.contracts.tools import ToolRuntime, ToolSpec


@dataclass(slots=True, frozen=True)
class _ToolChoice:
    """Normalized tool choice available to the tool-calling policy."""

    tool_name: str
    description: str
    input_schema: dict[str, object]
    default_tool_input: dict[str, object] | None = None
    keywords: tuple[str, ...] = ()


class ToolCallingAgent(Agent):
    """Agent that asks the model to select a tool and structured arguments."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        model: str = "gpt-4o-mini",
        default_tool_name: str = "text_stats_tool",
    ) -> None:
        """Initialize a tool-calling agent with injected runtime dependencies.

        Args:
            llm_client: LLM client used for prompt execution.
            tool_runtime: Tool runtime used for tool invocation.
            model: Default model name used for LLM calls.
            default_tool_name: Fallback tool used when no explicit choices are supplied.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._model = model
        self._default_tool_name = default_tool_name

    def run(self, input: Mapping[str, object], context: Mapping[str, object]) -> AgentResult:
        """Run one tool-calling step: plan tool call, then execute it."""
        prompt = _extract_prompt(input)
        tool_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        choices = _extract_tool_choices(
            input_payload=input,
            tool_specs=tool_specs,
            default_tool_name=self._default_tool_name,
        )

        model_messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a tool-calling planner. "
                    "Choose exactly one tool and arguments. "
                    "Return only JSON."
                ),
            ),
            LLMMessage(
                role="user",
                content=_build_tool_call_prompt(prompt=prompt, choices=choices),
            ),
        ]
        llm_params = LLMChatParams(
            response_schema=_tool_call_response_schema([choice.tool_name for choice in choices]),
            provider_options={"agent": "ToolCallingAgent"},
        )
        llm_response = self._llm_client.chat(model_messages, model=self._model, params=llm_params)

        parsed_tool_call = _parse_tool_call(llm_response.text)
        selected_choice, tool_call_source, tool_call_reason = _select_tool_choice(
            parsed_tool_call=parsed_tool_call,
            prompt=prompt,
            choices=choices,
        )
        tool_input = _resolve_tool_input(
            selected_choice=selected_choice,
            parsed_tool_call=parsed_tool_call,
            input_payload=input,
            context=context,
            llm_response_text=llm_response.text,
        )

        tool_result = self._tool_runtime.invoke(selected_choice.tool_name, tool_input, context)
        output: dict[str, object] = {
            "model_text": llm_response.text,
            "tool_name": selected_choice.tool_name,
            "tool_input": tool_input,
            "tool_output": tool_result.output,
        }
        return AgentResult(
            output=output,
            success=tool_result.success,
            tool_results=[tool_result],
            model_response=llm_response,
            metadata={
                "context_keys": sorted(context.keys()),
                "tool_call": {
                    "source": tool_call_source,
                    "reason": tool_call_reason,
                    "available_tools": [choice.tool_name for choice in choices],
                    "parsed_tool_call": parsed_tool_call,
                },
            },
        )

    def run_stream(
        self,
        input: Mapping[str, object],
        context: Mapping[str, object],
    ) -> Iterator[AgentStreamEvent]:
        """Stream a deterministic event pair around ``run``."""
        result = self.run(input, context)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from the input payload."""
    raw_prompt = input_payload.get(
        "prompt", input_payload.get("text", "Provide a concise response.")
    )
    return str(raw_prompt)


def _extract_tool_choices(
    *,
    input_payload: Mapping[str, object],
    tool_specs: Mapping[str, ToolSpec],
    default_tool_name: str,
) -> list[_ToolChoice]:
    """Extract normalized tool choices from input payload or runtime defaults."""
    raw_choices = input_payload.get("tools", input_payload.get("alternatives"))
    normalized = _normalize_explicit_tool_choices(raw_choices=raw_choices, tool_specs=tool_specs)
    if normalized:
        return normalized

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


def _normalize_explicit_tool_choices(
    *,
    raw_choices: object,
    tool_specs: Mapping[str, ToolSpec],
) -> list[_ToolChoice]:
    """Normalize explicit tool choices provided on the input payload."""
    if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, (str, bytes)):
        return []

    normalized: list[_ToolChoice] = []
    for raw_choice in raw_choices:
        if not isinstance(raw_choice, Mapping):
            continue
        raw_tool_name = raw_choice.get("tool_name", raw_choice.get("name"))
        if not isinstance(raw_tool_name, str):
            continue
        tool_name = raw_tool_name.strip()
        if not tool_name:
            continue

        runtime_spec = tool_specs.get(tool_name)
        raw_input_schema = raw_choice.get("input_schema")
        if isinstance(raw_input_schema, Mapping):
            input_schema = dict(raw_input_schema)
        elif runtime_spec is not None:
            input_schema = dict(runtime_spec.input_schema)
        else:
            input_schema = {"type": "object"}

        raw_description = raw_choice.get("description")
        if raw_description is None and runtime_spec is not None:
            description = runtime_spec.description
        else:
            description = str(raw_description or "")

        raw_default_tool_input = raw_choice.get("tool_input")
        default_tool_input = (
            dict(raw_default_tool_input) if isinstance(raw_default_tool_input, Mapping) else None
        )
        normalized.append(
            _ToolChoice(
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                default_tool_input=default_tool_input,
                keywords=_extract_keywords(raw_choice.get("keywords")),
            )
        )

    return normalized


def _extract_keywords(raw_keywords: object) -> tuple[str, ...]:
    """Normalize keyword hints for one tool choice."""
    if not isinstance(raw_keywords, Sequence) or isinstance(raw_keywords, (str, bytes)):
        return ()
    normalized: list[str] = []
    for raw_keyword in raw_keywords:
        if not isinstance(raw_keyword, str):
            continue
        keyword = raw_keyword.strip().lower()
        if keyword:
            normalized.append(keyword)
    return tuple(normalized)


def _build_tool_call_prompt(*, prompt: str, choices: Sequence[_ToolChoice]) -> str:
    """Build the model prompt for selecting a tool and arguments."""
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
    choices_text = "\n".join(choice_lines)
    return "\n".join(
        [
            "Select exactly one tool from the list and provide JSON arguments.",
            "Return only one JSON object with this shape:",
            '{"tool_name":"<name>","tool_input":{...},"reason":"short rationale"}',
            "Do not include markdown or extra text.",
            "",
            "Available tools:",
            choices_text,
            "",
            "User request:",
            prompt,
        ]
    )


def _tool_call_response_schema(tool_names: Sequence[str]) -> dict[str, object]:
    """Return JSON schema used to constrain tool-calling model output."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_name", "tool_input"],
        "properties": {
            "tool_name": {"type": "string", "enum": list(tool_names)},
            "tool_input": {"type": "object"},
            "reason": {"type": "string"},
        },
    }


def _parse_tool_call(raw_text: str) -> dict[str, object] | None:
    """Parse tool-call JSON from model text."""
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
    """Load text as a JSON object and return ``None`` when invalid."""
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
    """Select a validated tool choice from model output or fallback routing."""
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
    context: Mapping[str, object],
    llm_response_text: str,
) -> dict[str, object]:
    """Resolve the final tool input from model output and deterministic fallbacks."""
    if parsed_tool_call is not None:
        raw_tool_input = parsed_tool_call.get(
            "tool_input",
            parsed_tool_call.get("arguments", parsed_tool_call.get("args")),
        )
        normalized_from_model = _coerce_tool_input(raw_tool_input)
        if normalized_from_model:
            return normalized_from_model

    if selected_choice.default_tool_input is not None:
        return dict(selected_choice.default_tool_input)

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
        raw_dependency_results = context.get("dependency_results")
        if isinstance(raw_dependency_results, Mapping) and raw_dependency_results:
            return {"text": json.dumps(dict(raw_dependency_results), sort_keys=True)}
        return {"text": llm_response_text}

    return {}


def _coerce_tool_input(raw_tool_input: object) -> dict[str, object] | None:
    """Convert raw tool input into a JSON-like dictionary."""
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
    """Select a fallback tool choice using deterministic lexical signals."""
    prompt_text = prompt.lower()
    prompt_tokens = _tokenize(prompt_text)
    prompt_looks_math = _looks_like_arithmetic_request(prompt_text)
    prompt_looks_text = _looks_like_text_analysis_request(prompt_text)

    selected_choice = choices[0]
    selected_score = -1
    selected_reason = "fallback-first-choice"
    for choice in choices:
        searchable = " ".join([choice.tool_name, choice.description, *choice.keywords]).lower()
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
    """Infer a calculator expression from input payload and prompt text."""
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
    """Tokenize text into normalized alphanumeric words."""
    return {token for token in re.findall(r"[a-z0-9_]+", text) if token}


def _looks_like_arithmetic_request(text: str) -> bool:
    """Determine whether prompt text appears to request arithmetic."""
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
    """Determine whether prompt text appears to request text analysis."""
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
    """Determine whether choice text represents an arithmetic tool."""
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
    """Determine whether choice text represents a text-analysis tool."""
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
