"""Router agent implementation that selects one tool alternative per request."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMChatParams, LLMClient, LLMMessage
from design_research_agents.contracts.tools import ToolRuntime


@dataclass(slots=True, frozen=True)
class _ToolAlternative:
    """Normalized tool route candidate."""

    tool_name: str
    description: str
    tool_input: dict[str, object] | None = None
    keywords: tuple[str, ...] = ()


class RouterAgent(Agent):
    """Agent that routes one request to one selected tool alternative."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        model: str = "gpt-4o-mini",
        default_tool_name: str = "text_stats_tool",
    ) -> None:
        """Initialize a router agent with injected runtime dependencies.

        Args:
            llm_client: LLM client used for prompt execution.
            tool_runtime: Tool runtime used for tool invocation.
            model: Default model name used for LLM calls.
            default_tool_name: Fallback tool used when no alternatives are supplied.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._model = model
        self._default_tool_name = default_tool_name

    def run(self, input: Mapping[str, object], context: Mapping[str, object]) -> AgentResult:
        """Run one model call and one routed tool invocation."""
        prompt = _extract_prompt(input)
        response_schema = _extract_response_schema(input)
        alternatives = _extract_alternatives(
            input_payload=input,
            default_tool_name=self._default_tool_name,
        )
        selected_alternative, selected_index, selected_reason, scored_routes = _route_alternative(
            prompt=prompt,
            alternatives=alternatives,
        )

        llm_params = LLMChatParams(
            response_schema=response_schema,
            provider_options={"agent": "RouterAgent"},
        )
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a practical routing assistant. "
                    "Answer directly and avoid repeating prompt or schema text."
                ),
            ),
            LLMMessage(role="user", content=prompt),
        ]
        llm_response = self._llm_client.chat(messages, model=self._model, params=llm_params)

        tool_input = _resolve_tool_input(
            tool_name=selected_alternative.tool_name,
            explicit_tool_input=selected_alternative.tool_input,
            input_payload=input,
            context=context,
            llm_response_text=llm_response.text,
        )
        tool_result = self._tool_runtime.invoke(selected_alternative.tool_name, tool_input, context)

        output: dict[str, object] = {
            "model_text": llm_response.text,
            "tool_name": selected_alternative.tool_name,
            "selected_alternative_index": selected_index,
            "tool_output": tool_result.output,
        }
        return AgentResult(
            output=output,
            success=tool_result.success,
            tool_results=[tool_result],
            model_response=llm_response,
            metadata={
                "context_keys": sorted(context.keys()),
                "routing": {
                    "alternatives": [candidate.tool_name for candidate in alternatives],
                    "selected_tool_name": selected_alternative.tool_name,
                    "selected_alternative_index": selected_index,
                    "selected_reason": selected_reason,
                    "scored_routes": scored_routes,
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


def _extract_response_schema(
    input_payload: Mapping[str, object],
) -> dict[str, object] | None:
    """Extract an optional response schema from input payload."""
    raw_schema = input_payload.get("response_schema")
    if isinstance(raw_schema, dict):
        return dict(raw_schema)
    return None


def _extract_alternatives(
    *,
    input_payload: Mapping[str, object],
    default_tool_name: str,
) -> list[_ToolAlternative]:
    """Extract normalized routing alternatives from input payload."""
    raw_alternatives = input_payload.get("alternatives")
    normalized: list[_ToolAlternative] = []
    if isinstance(raw_alternatives, Sequence) and not isinstance(raw_alternatives, (str, bytes)):
        for raw_alternative in raw_alternatives:
            if not isinstance(raw_alternative, Mapping):
                continue
            raw_tool_name = raw_alternative.get("tool_name", raw_alternative.get("name"))
            if not isinstance(raw_tool_name, str):
                continue
            tool_name = raw_tool_name.strip()
            if not tool_name:
                continue
            raw_description = raw_alternative.get("description", "")
            raw_tool_input = raw_alternative.get("tool_input")
            normalized.append(
                _ToolAlternative(
                    tool_name=tool_name,
                    description=str(raw_description),
                    tool_input=(
                        dict(raw_tool_input) if isinstance(raw_tool_input, Mapping) else None
                    ),
                    keywords=_extract_keywords(raw_alternative.get("keywords")),
                )
            )

    if normalized:
        return normalized

    # Keep direct ``tool_name`` support to avoid breaking older callers.
    legacy_tool_name = input_payload.get("tool_name")
    if isinstance(legacy_tool_name, str) and legacy_tool_name.strip():
        return [
            _ToolAlternative(
                tool_name=legacy_tool_name.strip(),
                description="Legacy direct tool route.",
                tool_input=_coerce_tool_input(input_payload.get("tool_input")),
            )
        ]

    return [
        _ToolAlternative(
            tool_name=default_tool_name,
            description="Default fallback route.",
            tool_input=_coerce_tool_input(input_payload.get("tool_input")),
        )
    ]


def _extract_keywords(raw_keywords: object) -> tuple[str, ...]:
    """Normalize keyword hints for one alternative."""
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


def _coerce_tool_input(raw_tool_input: object) -> dict[str, object] | None:
    """Normalize optional tool input into a plain dictionary."""
    if isinstance(raw_tool_input, Mapping):
        return dict(raw_tool_input)
    return None


def _route_alternative(
    *,
    prompt: str,
    alternatives: Sequence[_ToolAlternative],
) -> tuple[_ToolAlternative, int, str, list[dict[str, object]]]:
    """Choose one alternative based on prompt/alternative overlap signals."""
    prompt_text = prompt.lower()
    prompt_tokens = _tokenize(prompt_text)
    prompt_looks_math = _looks_like_arithmetic_request(prompt_text)
    prompt_looks_text = _looks_like_text_analysis_request(prompt_text)

    selected_index = 0
    best_score = -1
    selected_reason = "fallback"
    scored_routes: list[dict[str, object]] = []
    for index, alternative in enumerate(alternatives):
        searchable = " ".join(
            [alternative.tool_name, alternative.description, *alternative.keywords]
        ).lower()
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

        reason = ", ".join(reason_parts) if reason_parts else "fallback"
        scored_routes.append(
            {
                "tool_name": alternative.tool_name,
                "score": score,
                "reason": reason,
            }
        )
        if score > best_score:
            selected_index = index
            best_score = score
            selected_reason = reason

    return alternatives[selected_index], selected_index, selected_reason, scored_routes


def _resolve_tool_input(
    *,
    tool_name: str,
    explicit_tool_input: Mapping[str, object] | None,
    input_payload: Mapping[str, object],
    context: Mapping[str, object],
    llm_response_text: str,
) -> dict[str, object]:
    """Resolve tool input from explicit payload or inferred defaults."""
    if explicit_tool_input is not None:
        return dict(explicit_tool_input)

    raw_tool_input = input_payload.get("tool_input")
    if isinstance(raw_tool_input, Mapping):
        return dict(raw_tool_input)

    if tool_name == "calculator_tool":
        expression = input_payload.get(
            "expression", input_payload.get("text", input_payload.get("prompt", ""))
        )
        return {"expression": str(expression)}

    if tool_name == "text_stats_tool":
        analysis_text = input_payload.get("analysis_text")
        if analysis_text is not None:
            return {"text": str(analysis_text)}
        raw_dependency_results = context.get("dependency_results")
        if isinstance(raw_dependency_results, Mapping) and raw_dependency_results:
            dependency_text = json.dumps(dict(raw_dependency_results), sort_keys=True)
            return {"text": dependency_text}
        return {"text": llm_response_text}

    return {}


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
    """Determine whether alternative text represents an arithmetic tool."""
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
    """Determine whether alternative text represents a text-analysis tool."""
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
