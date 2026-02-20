"""Reusable debate-pattern orchestration chunk."""

from __future__ import annotations

import json
from collections.abc import Mapping
from string import Template
from uuid import uuid4

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import LLMChatParams, LLMClient, LLMMessage, LLMResponse
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import LogicStep, LoopStep
from design_research_agents.implementations.agents.direct_llm_call import (
    DirectLLMCall,
)
from design_research_agents.implementations.shared.agent_internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.implementations.shared.agent_internal.model_resolution import (
    resolve_agent_model,
)
from design_research_agents.implementations.shared.agent_internal.prompt_overrides import (
    validate_prompt_text,
)
from design_research_agents.implementations.shared.agent_internal.result_builders import (
    build_failure_result,
)
from design_research_agents.schemas import (
    SchemaValidationError,
    validate_payload_against_schema,
)
from design_research_agents.tracing import Tracer
from design_research_agents.workflow import Workflow

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


class _DebateWorkflowCallbacks:
    """Workflow callback bundle used by debate round/judge steps."""

    def __init__(
        self,
        *,
        pattern: DebatePattern,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        affirmative_agent: DirectLLMCall,
        negative_agent: DirectLLMCall,
        runtime_state: dict[str, object],
    ) -> None:
        """Store dependencies used by callback methods.

        Args:
            pattern: Debate pattern instance.
            prompt: User task prompt.
            request_id: Resolved request id.
            dependencies: Resolved dependency mapping.
            affirmative_agent: Affirmative direct-call delegate.
            negative_agent: Negative direct-call delegate.
            runtime_state: Mutable state used to retain last model response.
        """
        self._pattern = pattern
        self._prompt = prompt
        self._request_id = request_id
        self._dependencies = dependencies
        self._affirmative_agent = affirmative_agent
        self._negative_agent = negative_agent
        self._runtime_state = runtime_state

    def continue_predicate(self, iteration: int, state: Mapping[str, object]) -> bool:
        """Return whether the debate loop should continue.

        Args:
            iteration: One-based loop iteration index.
            state: Current loop state mapping.

        Returns:
            ``True`` when debate rounds should continue.
        """
        del iteration
        return bool(state.get("should_continue", True))

    def run_round(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute one affirmative/negative round.

        Args:
            context: Workflow step context.

        Returns:
            Updated loop state for the next debate round.
        """
        loop_meta = context.get("_loop")
        round_number = 1
        if isinstance(loop_meta, Mapping):
            round_number = _safe_int(loop_meta.get("iteration", 1))
            if round_number < 1:
                round_number = 1

        loop_state = context.get("loop_state")
        current_state = dict(loop_state) if isinstance(loop_state, Mapping) else {}
        raw_rounds = current_state.get("rounds")
        rounds = (
            [dict(round_item) for round_item in raw_rounds if isinstance(round_item, Mapping)]
            if isinstance(raw_rounds, list)
            else []
        )
        prior_negative_argument = str(current_state.get("prior_negative_argument", "(none)"))

        affirmative_prompt = _render_prompt_template(
            template_text=self._pattern._affirmative_user_prompt_template,
            variables={
                "task_prompt": self._prompt,
                "round": round_number,
                "opponent_argument": prior_negative_argument,
            },
            field_name="debate_affirmative_user_prompt_template",
        )
        affirmative_result = self._affirmative_agent.run(
            affirmative_prompt,
            request_id=f"{self._request_id}:debate:affirmative:{round_number}",
            dependencies=self._dependencies,
        )
        if affirmative_result.model_response is not None:
            self._runtime_state["last_model_response"] = affirmative_result.model_response
        affirmative_argument = str(affirmative_result.output.get("model_text", "")).strip()

        negative_prompt = _render_prompt_template(
            template_text=self._pattern._negative_user_prompt_template,
            variables={
                "task_prompt": self._prompt,
                "round": round_number,
                "opponent_argument": affirmative_argument or "(none)",
            },
            field_name="debate_negative_user_prompt_template",
        )
        negative_result = self._negative_agent.run(
            negative_prompt,
            request_id=f"{self._request_id}:debate:negative:{round_number}",
            dependencies=self._dependencies,
        )
        if negative_result.model_response is not None:
            self._runtime_state["last_model_response"] = negative_result.model_response
        negative_argument = str(negative_result.output.get("model_text", "")).strip()
        rounds.append(
            {
                "round": round_number,
                "affirmative_argument": affirmative_argument,
                "negative_argument": negative_argument,
            }
        )
        return {
            "rounds": rounds,
            "prior_negative_argument": negative_argument or "(none)",
            "should_continue": True,
        }

    @staticmethod
    def state_reducer(
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        """Fold one debate loop iteration into accumulated state.

        Args:
            state: Current loop state mapping.
            iteration_result: Execution result for one loop iteration.
            iteration: One-based loop iteration index.

        Returns:
            Reduced loop state mapping.
        """
        del iteration
        iteration_step = iteration_result.step_results.get("debate_round")
        if iteration_step is None or not getattr(iteration_step, "success", False):
            return dict(state)
        output = getattr(iteration_step, "output", {})
        return dict(output) if isinstance(output, Mapping) else dict(state)

    def run_judge(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute the judge step after debate rounds complete.

        Args:
            context: Workflow step context.

        Returns:
            Judge output payload with status, error, rounds, and verdict.
        """
        dependency_results = context.get("dependency_results")
        rounds: list[dict[str, object]] = []
        if isinstance(dependency_results, Mapping):
            loop_step = dependency_results.get("debate_rounds")
            if isinstance(loop_step, Mapping):
                loop_output = loop_step.get("output")
                if isinstance(loop_output, Mapping):
                    final_state = loop_output.get("final_state")
                    if isinstance(final_state, Mapping):
                        raw_rounds = final_state.get("rounds")
                        if isinstance(raw_rounds, list):
                            rounds = [
                                dict(round_item)
                                for round_item in raw_rounds
                                if isinstance(round_item, Mapping)
                            ]

        resolved_model = resolve_agent_model(llm_client=self._pattern._llm_client)
        judge_messages = [
            LLMMessage(role="system", content=self._pattern._judge_system_prompt),
            LLMMessage(
                role="user",
                content=_render_prompt_template(
                    template_text=self._pattern._judge_user_prompt_template,
                    variables={
                        "task_prompt": self._prompt,
                        "debate_rounds_json": json.dumps(
                            rounds,
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                    },
                    field_name="debate_judge_user_prompt_template",
                ),
            ),
        ]
        judge_response = self._pattern._llm_client.chat(
            judge_messages,
            model=resolved_model,
            params=LLMChatParams(response_schema=dict(_VERDICT_SCHEMA)),
        )
        self._runtime_state["last_model_response"] = judge_response
        parsed_verdict = _parse_json_mapping(judge_response.text)
        if parsed_verdict is None:
            return {
                "status": "judge_invalid_json",
                "error": "Debate judge did not return valid JSON output.",
                "rounds": rounds,
                "verdict": None,
            }
        try:
            validate_payload_against_schema(
                payload=parsed_verdict,
                schema=_VERDICT_SCHEMA,
                location="debate_pattern.judge",
            )
        except SchemaValidationError as exc:
            return {
                "status": "judge_invalid_schema",
                "error": f"Debate judge output failed schema validation: {exc}",
                "rounds": rounds,
                "verdict": parsed_verdict,
            }
        return {
            "status": "completed",
            "error": None,
            "rounds": rounds,
            "verdict": parsed_verdict,
        }


class DebatePattern(Agent):
    """Configured reusable debate pattern with affirmative, negative, and judge phases."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        max_rounds: int = 3,
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
            max_rounds: Maximum number of debate rounds before judgment.
            debate_affirmative_system_prompt: Optional override for affirmative system prompt text.
            debate_affirmative_user_prompt_template: Optional override for affirmative template.
            debate_negative_system_prompt: Optional override for negative system prompt text.
            debate_negative_user_prompt_template: Optional override for negative user template.
            debate_judge_system_prompt: Optional override for judge system prompt text.
            debate_judge_user_prompt_template: Optional override for judge user prompt template.
            default_request_id_prefix: Optional prefix used when auto-generating request IDs.
            default_dependencies: Optional default dependency mapping merged into run calls.
            tracer: Optional tracer used by internal direct-call agents.

        Returns:
            None.

        Raises:
            ValueError: Raised when prompt overrides or request ID prefix are invalid.
        """
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")

        del tool_runtime
        self._llm_client = llm_client
        self._max_rounds = max_rounds
        self._default_request_id_prefix = _normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._tracer = tracer
        self.workflow: object | None = None
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
    ) -> ExecutionResult:
        """Run the debate pattern and return one final judged result.

        Args:
            prompt: Task prompt debated by affirmative and negative agents.
            request_id: Optional request ID override for trace and metadata correlation.
            dependencies: Optional run-scoped dependencies merged over default dependencies.

        Returns:
            Final ``ExecutionResult`` containing rounds, verdict payload, and synthesis output.

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
        affirmative_agent = DirectLLMCall(
            llm_client=self._llm_client,
            system_prompt=self._affirmative_system_prompt,
            tracer=self._tracer,
        )
        negative_agent = DirectLLMCall(
            llm_client=self._llm_client,
            system_prompt=self._negative_system_prompt,
            tracer=self._tracer,
        )
        runtime_state: dict[str, object] = {"last_model_response": None}
        callbacks = _DebateWorkflowCallbacks(
            pattern=self,
            prompt=prompt,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            affirmative_agent=affirmative_agent,
            negative_agent=negative_agent,
            runtime_state=runtime_state,
        )

        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_mode="schema",
            steps=[
                LoopStep(
                    step_id="debate_rounds",
                    steps=(LogicStep(step_id="debate_round", handler=callbacks.run_round),),
                    max_iterations=self._max_rounds,
                    initial_state={
                        "rounds": [],
                        "prior_negative_argument": "(none)",
                        "should_continue": True,
                    },
                    continue_predicate=callbacks.continue_predicate,
                    state_reducer=callbacks.state_reducer,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                ),
                LogicStep(
                    step_id="debate_judge",
                    dependencies=("debate_rounds",),
                    handler=callbacks.run_judge,
                ),
            ],
        )
        self.workflow = workflow
        workflow_result = workflow.run(
            {},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{resolved_request_id}:debate_workflow",
            dependencies=resolved_dependencies,
        )
        return _build_debate_result(
            workflow_result=workflow_result,
            runtime_state=runtime_state,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
        )


def _build_debate_result(
    *,
    workflow_result: ExecutionResult,
    runtime_state: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> ExecutionResult:
    """Build final debate output from workflow result payloads.

    Args:
        workflow_result: Completed debate workflow result.
        runtime_state: Runtime state carrying final model response.
        request_id: Resolved request id.
        dependencies: Resolved dependency mapping.

    Returns:
        Final normalized execution result for debate pattern runs.
    """
    judge_step = workflow_result.step_results.get("debate_judge")
    judge_output = judge_step.output if judge_step is not None else {}
    judge_status = str(judge_output.get("status", "judge_invalid_json"))
    rounds = judge_output.get("rounds")
    normalized_rounds = (
        [dict(round_item) for round_item in rounds if isinstance(round_item, Mapping)]
        if isinstance(rounds, list)
        else []
    )
    parsed_verdict = judge_output.get("verdict")
    normalized_verdict = dict(parsed_verdict) if isinstance(parsed_verdict, Mapping) else None
    workflow_payload = workflow_result.asdict()
    workflow_artifacts = workflow_result.output.get("artifacts", [])
    last_model_response = runtime_state.get("last_model_response")
    model_response = last_model_response if isinstance(last_model_response, LLMResponse) else None

    if judge_status != "completed" or normalized_verdict is None:
        terminated_reason = (
            judge_status
            if judge_status in {"judge_invalid_json", "judge_invalid_schema"}
            else "judge_invalid_json"
        )
        error_text = str(
            judge_output.get("error", "Debate judge did not return valid JSON output.")
        )
        return build_failure_result(
            error=error_text,
            model_response=model_response,
            tool_results=[],
            request_id=request_id,
            dependencies=dependencies,
            metadata={"mode": "debate_pattern", "stage": "judge"},
            output={
                "terminated_reason": terminated_reason,
                "rounds": normalized_rounds,
                "verdict": normalized_verdict,
                "final_output": {},
                "workflow": workflow_payload,
                "artifacts": workflow_artifacts,
            },
        )

    synthesis = str(normalized_verdict.get("synthesis", "")).strip()
    return ExecutionResult(
        output={
            "rounds": normalized_rounds,
            "verdict": normalized_verdict,
            "winner": normalized_verdict.get("winner", "tie"),
            "final_output": {"synthesis": synthesis},
            "terminated_reason": "completed",
            "workflow": workflow_payload,
            "artifacts": workflow_artifacts,
        },
        success=True,
        tool_results=[],
        model_response=model_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            "mode": "debate_pattern",
            "rounds": len(normalized_rounds),
        },
    )


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


def _safe_int(value: object) -> int:
    """Convert values to int with deterministic fallback to one.

    Args:
        value: Value to normalize into an integer.

    Returns:
        Integer representation, or ``1`` when conversion is not possible.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 1
    return 1


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
