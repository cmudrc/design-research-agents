"""Reusable debate-pattern orchestration chunk."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from string import Template
from uuid import uuid4

from design_research_agents.agent.implementations.single_step_direct_llm_agent import (
    SingleStepDirectLLMAgent,
)
from design_research_agents.agent.internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.prompt_overrides import validate_prompt_text
from design_research_agents.agent.internal.result_builders import build_failure_result
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMChatParams, LLMClient, LLMMessage
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.schemas import SchemaValidationError, validate_payload_against_schema
from design_research_agents.tracing import Tracer

_VERDICT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["winner", "rationale", "synthesis"],
    "properties": {
        "winner": {"type": "string", "enum": ["affirmative", "negative", "tie"]},
        "rationale": {"type": "string"},
        "synthesis": {"type": "string"},
    },
}

_DEFAULT_AFFIRMATIVE_SYSTEM_PROMPT = (
    "You are the affirmative side in a structured debate. "
    "Argue for the strongest case in favor of the task."
)
_DEFAULT_AFFIRMATIVE_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Round: $round",
        "Opponent argument from prior round:",
        "$opponent_argument",
        "Respond with a concise affirmative argument only.",
    ]
)
_DEFAULT_NEGATIVE_SYSTEM_PROMPT = (
    "You are the negative side in a structured debate. "
    "Argue the strongest case against the task's affirmative position."
)
_DEFAULT_NEGATIVE_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Round: $round",
        "Opponent argument this round:",
        "$opponent_argument",
        "Respond with a concise negative argument only.",
    ]
)
_DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict debate judge. Return JSON only with winner, rationale, and synthesis."
)
_DEFAULT_JUDGE_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task:",
        "$task_prompt",
        "",
        "Debate rounds (JSON):",
        "$debate_rounds_json",
        "",
        "Pick a winner and provide a concise synthesis.",
    ]
)


class DebatePattern(Agent):
    """Configured reusable debate pattern with affirmative, negative, and judge phases."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        controls: RuntimeControls | None = None,
        debate_affirmative_system_prompt: str | None = None,
        debate_affirmative_user_prompt_template: str | None = None,
        debate_negative_system_prompt: str | None = None,
        debate_negative_user_prompt_template: str | None = None,
        debate_judge_system_prompt: str | None = None,
        debate_judge_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize prompt defaults.

        Args:
            llm_client: Client used for affirmative, negative, and judge model calls.
            tool_runtime: Tool runtime dependency for ``Agent`` interface compatibility.
            controls: Runtime controls that define iteration and streaming behavior.
            debate_affirmative_system_prompt: Optional override for affirmative system prompt text.
            debate_affirmative_user_prompt_template: Optional override for affirmative template.
            debate_negative_system_prompt: Optional override for negative system prompt text.
            debate_negative_user_prompt_template: Optional override for negative user template.
            debate_judge_system_prompt: Optional override for judge system prompt text.
            debate_judge_user_prompt_template: Optional override for judge user prompt template.
            default_request_id_prefix: Optional prefix used when auto-generating request IDs.
            default_dependencies: Optional default dependency mapping merged into run calls.
            tracer: Optional tracer used by internal single-step agents.

        Returns:
            None.

        Raises:
            ValueError: Raised when prompt overrides or request ID prefix are invalid.
        """
        del tool_runtime
        self._llm_client = llm_client
        self._controls = controls or RuntimeControls()
        self._default_request_id_prefix = _normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._tracer = tracer
        self._affirmative_system_prompt = _resolve_prompt_override(
            override=debate_affirmative_system_prompt,
            default_value=_DEFAULT_AFFIRMATIVE_SYSTEM_PROMPT,
            field_name="debate_affirmative_system_prompt",
        )
        self._affirmative_user_prompt_template = _resolve_prompt_override(
            override=debate_affirmative_user_prompt_template,
            default_value=_DEFAULT_AFFIRMATIVE_USER_PROMPT_TEMPLATE,
            field_name="debate_affirmative_user_prompt_template",
        )
        self._negative_system_prompt = _resolve_prompt_override(
            override=debate_negative_system_prompt,
            default_value=_DEFAULT_NEGATIVE_SYSTEM_PROMPT,
            field_name="debate_negative_system_prompt",
        )
        self._negative_user_prompt_template = _resolve_prompt_override(
            override=debate_negative_user_prompt_template,
            default_value=_DEFAULT_NEGATIVE_USER_PROMPT_TEMPLATE,
            field_name="debate_negative_user_prompt_template",
        )
        self._judge_system_prompt = _resolve_prompt_override(
            override=debate_judge_system_prompt,
            default_value=_DEFAULT_JUDGE_SYSTEM_PROMPT,
            field_name="debate_judge_system_prompt",
        )
        self._judge_user_prompt_template = _resolve_prompt_override(
            override=debate_judge_user_prompt_template,
            default_value=_DEFAULT_JUDGE_USER_PROMPT_TEMPLATE,
            field_name="debate_judge_user_prompt_template",
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run the debate pattern and return one final judged result.

        Args:
            prompt: Task prompt debated by affirmative and negative agents.
            request_id: Optional request ID override for trace and metadata correlation.
            dependencies: Optional run-scoped dependencies merged over default dependencies.

        Returns:
            Final ``AgentResult`` containing rounds, verdict payload, and synthesis output.

        Raises:
            ValueError: Raised when configured prompt templates are invalid
                or missing required variables.
        """
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        if resolved_request_id is None:
            resolved_request_id = f"debate:{uuid4().hex}"
        resolved_dependencies = _merge_dependencies(
            default_dependencies=self._default_dependencies,
            run_dependencies=dependencies,
        )

        affirmative_agent = SingleStepDirectLLMAgent(
            llm_client=self._llm_client,
            system_prompt=self._affirmative_system_prompt,
            tracer=self._tracer,
        )
        negative_agent = SingleStepDirectLLMAgent(
            llm_client=self._llm_client,
            system_prompt=self._negative_system_prompt,
            tracer=self._tracer,
        )

        rounds: list[dict[str, object]] = []
        prior_negative_argument = "(none)"
        last_model_response = None
        for round_index in range(self._controls.max_iterations):
            affirmative_prompt = _render_prompt_template(
                template_text=self._affirmative_user_prompt_template,
                variables={
                    "task_prompt": prompt,
                    "round": round_index + 1,
                    "opponent_argument": prior_negative_argument,
                },
                field_name="debate_affirmative_user_prompt_template",
            )
            affirmative_result = affirmative_agent.run(
                affirmative_prompt,
                request_id=f"{resolved_request_id}:debate:affirmative:{round_index + 1}",
                dependencies=resolved_dependencies,
            )
            if affirmative_result.model_response is not None:
                last_model_response = affirmative_result.model_response
            affirmative_argument = str(affirmative_result.output.get("model_text", "")).strip()

            negative_prompt = _render_prompt_template(
                template_text=self._negative_user_prompt_template,
                variables={
                    "task_prompt": prompt,
                    "round": round_index + 1,
                    "opponent_argument": affirmative_argument or "(none)",
                },
                field_name="debate_negative_user_prompt_template",
            )
            negative_result = negative_agent.run(
                negative_prompt,
                request_id=f"{resolved_request_id}:debate:negative:{round_index + 1}",
                dependencies=resolved_dependencies,
            )
            if negative_result.model_response is not None:
                last_model_response = negative_result.model_response
            negative_argument = str(negative_result.output.get("model_text", "")).strip()

            rounds.append(
                {
                    "round": round_index + 1,
                    "affirmative_argument": affirmative_argument,
                    "negative_argument": negative_argument,
                }
            )
            prior_negative_argument = negative_argument or "(none)"

        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        judge_messages = [
            LLMMessage(role="system", content=self._judge_system_prompt),
            LLMMessage(
                role="user",
                content=_render_prompt_template(
                    template_text=self._judge_user_prompt_template,
                    variables={
                        "task_prompt": prompt,
                        "debate_rounds_json": json.dumps(rounds, ensure_ascii=True, sort_keys=True),
                    },
                    field_name="debate_judge_user_prompt_template",
                ),
            ),
        ]
        judge_response = self._llm_client.chat(
            judge_messages,
            model=resolved_model,
            params=LLMChatParams(response_schema=dict(_VERDICT_SCHEMA)),
        )
        last_model_response = judge_response

        parsed_verdict = _parse_json_mapping(judge_response.text)
        if parsed_verdict is None:
            return build_failure_result(
                error="Debate judge did not return valid JSON output.",
                model_response=judge_response,
                tool_results=[],
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                metadata={"mode": "debate_pattern", "stage": "judge"},
                output={
                    "terminated_reason": "judge_invalid_json",
                    "rounds": rounds,
                    "verdict": None,
                },
            )

        try:
            validate_payload_against_schema(
                payload=parsed_verdict,
                schema=_VERDICT_SCHEMA,
                location="debate_pattern.judge",
            )
        except SchemaValidationError as exc:
            return build_failure_result(
                error=f"Debate judge output failed schema validation: {exc}",
                model_response=judge_response,
                tool_results=[],
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                metadata={"mode": "debate_pattern", "stage": "judge"},
                output={
                    "terminated_reason": "judge_invalid_schema",
                    "rounds": rounds,
                    "verdict": parsed_verdict,
                },
            )

        synthesis = str(parsed_verdict.get("synthesis", "")).strip()
        return AgentResult(
            output={
                "rounds": rounds,
                "verdict": parsed_verdict,
                "winner": parsed_verdict.get("winner", "tie"),
                "final_output": {"synthesis": synthesis},
                "terminated_reason": "completed",
            },
            success=True,
            tool_results=[],
            model_response=last_model_response,
            metadata={
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "mode": "debate_pattern",
                "rounds": len(rounds),
            },
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Run one debate and emit stream events.

        Args:
            prompt: Task prompt debated by affirmative and negative agents.
            request_id: Optional request ID override for trace and metadata correlation.
            dependencies: Optional run-scoped dependencies merged over default dependencies.

        Yields:
            ``AgentStreamEvent`` values containing optional delta text followed by completion.

        Raises:
            ValueError: Raised when configured prompt templates are invalid
                or missing required variables.
        """
        result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        if self._controls.streaming_enabled:
            delta_text = result.model_response.text if result.model_response is not None else ""
            yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)


def _merge_dependencies(
    *,
    default_dependencies: Mapping[str, object],
    run_dependencies: Mapping[str, object] | None,
) -> dict[str, object]:
    """Merge default dependencies with optional run-level dependency overrides.

    Args:
        default_dependencies: Default dependency mapping configured on the workflow instance.
        run_dependencies: Optional run-scoped dependency overrides.

    Returns:
        Merged dependency mapping where run-level values override defaults.
    """
    merged = dict(default_dependencies)
    if run_dependencies is not None:
        merged.update(run_dependencies)
    return merged


def _normalize_request_id_prefix(default_request_id_prefix: str | None) -> str | None:
    """Validate and normalize the default request ID prefix.

    Args:
        default_request_id_prefix: Optional request ID prefix configured by callers.

    Returns:
        Stripped non-empty prefix, or ``None`` when no prefix is configured.

    Raises:
        ValueError: Raised when a provided prefix is empty after stripping whitespace.
    """
    if default_request_id_prefix is None:
        return None
    normalized = default_request_id_prefix.strip()
    if not normalized:
        raise ValueError("default_request_id_prefix must be non-empty when provided.")
    return normalized


def _resolve_request_id(*, request_id: str | None, default_prefix: str | None) -> str | None:
    """Resolve request ID using an explicit value or configured default prefix.

    Args:
        request_id: Optional explicit request ID supplied for one run.
        default_prefix: Optional default prefix used to auto-generate a request ID.

    Returns:
        Explicit request ID when non-empty, generated ID when prefix
        is configured, otherwise ``None``.
    """
    if request_id is not None and request_id.strip():
        return request_id
    if default_prefix is None:
        return request_id
    return f"{default_prefix}:{uuid4().hex}"


def _resolve_prompt_override(
    *,
    override: str | None,
    default_value: str,
    field_name: str,
) -> str:
    """Resolve prompt text from override-or-default and validate it.

    Args:
        override: Optional override prompt text.
        default_value: Default prompt text used when no override is provided.
        field_name: Configuration field name used in validation error messages.

    Returns:
        Validated prompt text.

    Raises:
        ValueError: Raised when resolved prompt text is invalid.
    """
    if override is None:
        return validate_prompt_text(value=default_value, field_name=field_name)
    return validate_prompt_text(value=override, field_name=field_name)


def _render_prompt_template(
    *,
    template_text: str,
    variables: Mapping[str, object],
    field_name: str,
) -> str:
    """Validate and render one prompt template using string substitution variables.

    Args:
        template_text: Prompt template text that may reference ``$``-prefixed variables.
        variables: Mapping of variable names to values converted to strings before substitution.
        field_name: Configuration field name used in validation and substitution errors.

    Returns:
        Rendered prompt string.

    Raises:
        ValueError: Raised when template text is invalid or references missing variables.
    """
    normalized_template = validate_prompt_text(value=template_text, field_name=field_name)
    template = Template(normalized_template)
    rendered_variables = {key: str(value) for key, value in variables.items()}
    try:
        return template.substitute(rendered_variables)
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "unknown"
        raise ValueError(f"{field_name} is missing required variable '{missing_key}'.") from exc


__all__ = [
    "DebatePattern",
]
