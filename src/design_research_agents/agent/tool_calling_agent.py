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
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMChatParams, LLMClient, LLMMessage
from design_research_agents.contracts.tools import ToolRuntime, ToolSpec
from design_research_agents.prompts import load_prompt, render_prompt


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
            model: Optional model override applied when ``input['model']`` is absent.
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

    def run(self, input: Mapping[str, object], context: Mapping[str, object]) -> AgentResult:
        """Run one tool-calling step from planning through tool execution.

        The run prompts for a structured tool call, validates selection, resolves
        tool input, executes the tool, and returns unified output/metadata.
        """
        prompt = _extract_prompt(input)
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
            input_payload=input,
            init_model=self._model,
        )
        choices = [_clone_tool_choice(choice) for choice in self._compiled_tool_choices]
        alternatives_prompt_target = resolve_alternatives_prompt_target(input_payload=input)
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
        llm_params = LLMChatParams(
            response_schema=clone_response_schema(self._default_tool_call_response_schema),
            provider_options={"agent": "ToolCallingAgent"},
        )
        llm_response = self._llm_client.chat(
            model_messages,
            model=resolved_model,
            params=llm_params,
        )

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
        """Emit a deterministic stream wrapper around ``run``.

        The wrapper emits one delta containing full model text, followed by a
        completion event with the final ``AgentResult``.
        """
        result = self.run(input, context)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract request prompt text from input payload.

    Falls back to ``text`` and then a stable default string when absent.
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
    """Extract normalized tool choices from runtime specs."""
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
    """
    return render_prompt(
        "tool_calling_user_select_tool",
        variables={
            "choices_block": choices_block,
            "user_prompt": prompt,
        },
    )


def _build_tool_choices_text(*, choices: Sequence[_ToolChoice]) -> str:
    """Build formatted runtime tool choices text."""
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


def _tool_call_response_schema(tool_names: Sequence[str]) -> dict[str, object]:
    """Build JSON schema constraining tool-calling model output payloads.

    Restricts ``tool_name`` to currently available choices.
    """
    return build_tool_call_response_schema(tool_names=tool_names)


def _clone_tool_choice(choice: _ToolChoice) -> _ToolChoice:
    """Clone one tool choice so run-local payloads remain isolated."""
    return _ToolChoice(
        tool_name=choice.tool_name,
        description=choice.description,
        input_schema=dict(choice.input_schema),
    )


def _parse_tool_call(raw_text: str) -> dict[str, object] | None:
    """Parse tool-call JSON payload from model text output.

    Supports strict JSON responses and JSON objects embedded in surrounding text.
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
    """Load text as JSON object mapping and return ``None`` on invalid input.

    Non-object JSON payloads are treated as invalid for tool-call parsing.
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
    context: Mapping[str, object],
    llm_response_text: str,
) -> dict[str, object]:
    """Resolve final tool input from model payload, run input, or heuristics."""
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
        raw_dependency_results = context.get("dependency_results")
        if isinstance(raw_dependency_results, Mapping) and raw_dependency_results:
            return {"text": json.dumps(dict(raw_dependency_results), sort_keys=True)}
        return {"text": llm_response_text}

    return {}


def _coerce_tool_input(raw_tool_input: object) -> dict[str, object] | None:
    """Convert raw tool-input payload into a JSON-like dictionary when possible.

    Supports direct mappings and JSON-encoded string payloads.
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
    """Infer a calculator expression from payload fields and prompt text.

    Explicit payload expressions win; otherwise a regex candidate is extracted
    from prompt text before falling back to full prompt.
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
    """
    return {token for token in re.findall(r"[a-z0-9_]+", text) if token}


def _looks_like_arithmetic_request(text: str) -> bool:
    """Return whether prompt text appears to request arithmetic computation.

    Uses regex patterns and keyword heuristics.
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
