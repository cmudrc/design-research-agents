"""Reusable debate-pattern orchestration chunk."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import suppress

from design_research_agents._contracts._agent import Agent, ExecutionResult
from design_research_agents._contracts._llm import LLMClient, LLMMessage, LLMRequest, LLMResponse
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._contracts._workflow import (
    AgentStep,
    DelegateBatchCall,
    DelegateBatchStep,
    LogicStep,
    LoopStep,
    ModelStep,
    WorkflowDelegate,
)
from design_research_agents._implementations._agents._direct_llm_call import DirectLLMCall
from design_research_agents._implementations._shared._agent_internal._input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents._implementations._shared._agent_internal._model_resolution import (
    resolve_agent_model,
)
from design_research_agents._implementations._shared._agent_internal._result_builders import (
    build_failure_result,
)
from design_research_agents._runtime._patterns import (
    execute_pattern_with_trace,
    normalize_request_id_prefix,
    render_prompt_template,
    resolve_pattern_run_context,
    resolve_prompt_override,
)
from design_research_agents._runtime._patterns import (
    extract_call_error as _runtime_extract_call_error,
)
from design_research_agents._runtime._patterns import (
    extract_call_model_response as _runtime_extract_call_model_response,
)
from design_research_agents._runtime._patterns import (
    extract_call_output as _runtime_extract_call_output,
)
from design_research_agents._runtime._patterns import (
    extract_delegate_batch_call_result_from_context as _runtime_extract_delegate_batch_call_result_from_context,
)
from design_research_agents._runtime._patterns import (
    is_call_success as _runtime_is_call_success,
)
from design_research_agents._schemas import (
    SchemaValidationError,
    validate_payload_against_schema,
)
from design_research_agents._tracing import Tracer
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
    "You are the affirmative side in a structured debate. Argue for the strongest case in favor of the task."
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
_DEFAULT_JUDGE_SYSTEM_PROMPT = "You are a strict debate judge. Return JSON only with winner, rationale, and synthesis."
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
    """Workflow callback bundle used by debate round and judge steps."""

    def __init__(
        self,
        *,
        pattern: DebatePattern,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        affirmative_delegate: WorkflowDelegate,
        negative_delegate: WorkflowDelegate,
        judge_delegate: WorkflowDelegate | None,
        resolved_model: str,
        runtime_state: dict[str, object],
    ) -> None:
        """Store per-run callback dependencies."""
        self._pattern = pattern
        self._prompt = prompt
        self._request_id = request_id
        self._dependencies = dependencies
        self._affirmative_delegate = affirmative_delegate
        self._negative_delegate = negative_delegate
        self._judge_delegate = judge_delegate
        self._resolved_model = resolved_model
        self._runtime_state = runtime_state

    def continue_predicate(self, iteration: int, state: Mapping[str, object]) -> bool:
        """Return whether the debate loop should continue."""
        del iteration
        if not bool(state.get("should_continue", True)):
            return False
        failure_reason = state.get("failure_reason")
        return not (isinstance(failure_reason, str) and failure_reason)

    def build_affirmative_calls(
        self,
        context: Mapping[str, object],
    ) -> list[DelegateBatchCall]:
        """Build affirmative delegate call for one debate round."""
        round_number, _rounds, _prior_aff, prior_negative = _resolve_round_context(context)
        affirmative_prompt = render_prompt_template(
            template_text=self._pattern._affirmative_user_prompt_template,
            variables={
                "task_prompt": self._prompt,
                "round": round_number,
                "opponent_argument": prior_negative,
            },
            field_name="affirmative_user_prompt_template",
        )
        return [
            DelegateBatchCall(
                call_id="affirmative",
                delegate=self._affirmative_delegate,
                prompt=affirmative_prompt,
            )
        ]

    def build_negative_calls(
        self,
        context: Mapping[str, object],
    ) -> list[DelegateBatchCall]:
        """Build negative delegate call for one debate round."""
        round_number, _rounds, _prior_aff, _prior_negative = _resolve_round_context(context)
        affirmative_result = _extract_delegate_batch_call_result(
            context=context,
            dependency_step_id="debate_affirmative_batch",
            call_id="affirmative",
        )
        affirmative_output = _extract_call_output(affirmative_result)
        affirmative_argument = _extract_model_text_from_output(affirmative_output)
        negative_prompt = render_prompt_template(
            template_text=self._pattern._negative_user_prompt_template,
            variables={
                "task_prompt": self._prompt,
                "round": round_number,
                "opponent_argument": affirmative_argument or "(none)",
            },
            field_name="negative_user_prompt_template",
        )
        return [
            DelegateBatchCall(
                call_id="negative",
                delegate=self._negative_delegate,
                prompt=negative_prompt,
            )
        ]

    def build_round_state(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Build next loop state from affirmative and negative batch results."""
        round_number, rounds, _prior_affirmative, prior_negative = _resolve_round_context(context)

        affirmative_result = _extract_delegate_batch_call_result(
            context=context,
            dependency_step_id="debate_affirmative_batch",
            call_id="affirmative",
        )
        affirmative_response = _extract_call_model_response(affirmative_result)
        if affirmative_response is not None:
            self._runtime_state["last_model_response"] = affirmative_response
        if not _is_call_success(affirmative_result):
            return {
                "rounds": rounds,
                "prior_negative_argument": prior_negative,
                "prior_affirmative_argument": "(none)",
                "should_continue": False,
                "failure_reason": "affirmative_failed",
                "failure_error": _extract_call_error(
                    affirmative_result,
                    fallback_message="Affirmative delegate failed.",
                ),
            }
        affirmative_output = _extract_call_output(affirmative_result)
        affirmative_argument = _extract_model_text_from_output(affirmative_output)

        negative_result = _extract_delegate_batch_call_result(
            context=context,
            dependency_step_id="debate_negative_batch",
            call_id="negative",
        )
        negative_response = _extract_call_model_response(negative_result)
        if negative_response is not None:
            self._runtime_state["last_model_response"] = negative_response
        if not _is_call_success(negative_result):
            return {
                "rounds": rounds,
                "prior_negative_argument": prior_negative,
                "prior_affirmative_argument": affirmative_argument or "(none)",
                "should_continue": False,
                "failure_reason": "negative_failed",
                "failure_error": _extract_call_error(
                    negative_result,
                    fallback_message="Negative delegate failed.",
                ),
            }
        negative_output = _extract_call_output(negative_result)
        negative_argument = _extract_model_text_from_output(negative_output)

        rounds.append(
            {
                "round": round_number,
                "affirmative_argument": affirmative_argument,
                "negative_argument": negative_argument,
            }
        )
        return {
            "rounds": rounds,
            "prior_affirmative_argument": affirmative_argument or "(none)",
            "prior_negative_argument": negative_argument or "(none)",
            "should_continue": True,
            "failure_reason": None,
            "failure_error": None,
        }

    @staticmethod
    def state_reducer(
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        """Fold one debate round iteration into accumulated loop state."""
        del iteration
        iteration_step = iteration_result.step_results.get("debate_round")
        if iteration_step is None or not getattr(iteration_step, "success", False):
            return dict(state)
        output = getattr(iteration_step, "output", {})
        return dict(output) if isinstance(output, Mapping) else dict(state)

    def build_judge_prompt_from_context(self, context: Mapping[str, object]) -> str:
        """Build judge prompt from loop dependency results."""
        rounds = _extract_rounds_from_context(context)
        return _render_judge_prompt(
            prompt_template=self._pattern._judge_user_prompt_template,
            task_prompt=self._prompt,
            rounds=rounds,
        )

    def build_judge_request(self, context: Mapping[str, object]) -> LLMRequest:
        """Build model request for direct judge invocation."""
        judge_prompt = self.build_judge_prompt_from_context(context)
        judge_messages = [
            LLMMessage(role="system", content=self._pattern._judge_system_prompt),
            LLMMessage(role="user", content=judge_prompt),
        ]
        return LLMRequest(
            messages=judge_messages,
            model=self._resolved_model,
            response_schema=dict(_VERDICT_SCHEMA),
            metadata={
                "agent": "DebatePattern",
                "mode": "debate",
                "phase": "judge",
                "request_id": self._request_id,
            },
            provider_options={
                "agent": "DebatePattern",
                "mode": "debate",
                "phase": "judge",
            },
        )

    def parse_judge_response(
        self,
        response: LLMResponse,
        context: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Parse model judge response into structured verdict payload."""
        del context
        return {"verdict": _parse_json_mapping(response.text)}

    def build_judge_output_from_model(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Build normalized judge output from direct-model judge step payload."""
        rounds = _extract_rounds_from_context(context)
        judge_step_output = _extract_dependency_output(context, dependency_id="debate_judge_model")
        model_response = _extract_model_response_from_model_step_output(judge_step_output)
        if model_response is not None:
            self._runtime_state["last_model_response"] = model_response

        parsed_payload = judge_step_output.get("parsed")
        parsed_mapping = dict(parsed_payload) if isinstance(parsed_payload, Mapping) else {}
        parsed_verdict = parsed_mapping.get("verdict")
        if not isinstance(parsed_verdict, Mapping):
            return {
                "status": "judge_invalid_json",
                "error": "Debate judge did not return valid JSON output.",
                "rounds": rounds,
                "verdict": None,
            }

        normalized_verdict = dict(parsed_verdict)
        try:
            validate_payload_against_schema(
                payload=normalized_verdict,
                schema=_VERDICT_SCHEMA,
                location="debate_pattern.judge",
            )
        except SchemaValidationError as exc:
            return {
                "status": "judge_invalid_schema",
                "error": f"Debate judge output failed schema validation: {exc}",
                "rounds": rounds,
                "verdict": normalized_verdict,
            }
        return {
            "status": "completed",
            "error": None,
            "rounds": rounds,
            "verdict": normalized_verdict,
        }

    def build_judge_output_from_delegate(
        self,
        context: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Build normalized judge output from delegate judge step payload."""
        rounds = _extract_rounds_from_context(context)
        delegate_payload = _extract_dependency_output(
            context,
            dependency_id="debate_judge_delegate",
        )
        maybe_model_response = delegate_payload.get("model_response")
        if isinstance(maybe_model_response, Mapping):
            with suppress(TypeError):
                self._runtime_state["last_model_response"] = LLMResponse(**dict(maybe_model_response))

        delegate_success = delegate_payload.get("success")
        delegate_output_payload = delegate_payload.get("output")
        delegate_output = dict(delegate_output_payload) if isinstance(delegate_output_payload, Mapping) else {}
        if delegate_success is not True:
            error_text = str(delegate_output.get("error", "Debate judge delegate failed."))
            return {
                "status": "judge_invalid_json",
                "error": error_text,
                "rounds": rounds,
                "verdict": None,
            }

        parsed_verdict = _extract_delegate_verdict(delegate_output)
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
        affirmative_delegate: WorkflowDelegate | None = None,
        negative_delegate: WorkflowDelegate | None = None,
        judge_delegate: WorkflowDelegate | None = None,
        max_rounds: int = 3,
        affirmative_system_prompt: str | None = None,
        affirmative_user_prompt_template: str | None = None,
        negative_system_prompt: str | None = None,
        negative_user_prompt_template: str | None = None,
        judge_system_prompt: str | None = None,
        judge_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = "debate",
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize prompt defaults."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._max_rounds = max_rounds
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._tracer = tracer
        self.workflow: Workflow | None = None
        self._affirmative_delegate = affirmative_delegate
        self._negative_delegate = negative_delegate
        self._judge_delegate = judge_delegate
        self._affirmative_system_prompt = resolve_prompt_override(
            override=affirmative_system_prompt,
            default_value=_DEFAULT_AFFIRMATIVE_SYSTEM_PROMPT,
            field_name="affirmative_system_prompt",
        )
        self._affirmative_user_prompt_template = resolve_prompt_override(
            override=affirmative_user_prompt_template,
            default_value=_DEFAULT_AFFIRMATIVE_USER_PROMPT_TEMPLATE,
            field_name="affirmative_user_prompt_template",
        )
        self._negative_system_prompt = resolve_prompt_override(
            override=negative_system_prompt,
            default_value=_DEFAULT_NEGATIVE_SYSTEM_PROMPT,
            field_name="negative_system_prompt",
        )
        self._negative_user_prompt_template = resolve_prompt_override(
            override=negative_user_prompt_template,
            default_value=_DEFAULT_NEGATIVE_USER_PROMPT_TEMPLATE,
            field_name="negative_user_prompt_template",
        )
        self._judge_system_prompt = resolve_prompt_override(
            override=judge_system_prompt,
            default_value=_DEFAULT_JUDGE_SYSTEM_PROMPT,
            field_name="judge_system_prompt",
        )
        self._judge_user_prompt_template = resolve_prompt_override(
            override=judge_user_prompt_template,
            default_value=_DEFAULT_JUDGE_USER_PROMPT_TEMPLATE,
            field_name="judge_user_prompt_template",
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run the debate pattern and return one final judged result."""
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=self._default_request_id_prefix,
            default_dependencies=self._default_dependencies,
            request_id=request_id,
            dependencies=dependencies,
        )
        return execute_pattern_with_trace(
            agent_name="DebatePattern",
            request_id=run_context.request_id,
            input_payload={"prompt": prompt, "max_rounds": self._max_rounds, "mode": "debate"},
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            runner=lambda: self._run_debate(
                prompt=prompt,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _run_debate(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Execute debate rounds then judge verdict."""
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        affirmative_delegate = self._affirmative_delegate
        if affirmative_delegate is None:
            affirmative_delegate = DirectLLMCall(
                llm_client=self._llm_client,
                system_prompt=self._affirmative_system_prompt,
                tracer=self._tracer,
            )
        negative_delegate = self._negative_delegate
        if negative_delegate is None:
            negative_delegate = DirectLLMCall(
                llm_client=self._llm_client,
                system_prompt=self._negative_system_prompt,
                tracer=self._tracer,
            )

        runtime_state: dict[str, object] = {"last_model_response": None}
        callbacks = _DebateWorkflowCallbacks(
            pattern=self,
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            affirmative_delegate=affirmative_delegate,
            negative_delegate=negative_delegate,
            judge_delegate=self._judge_delegate,
            resolved_model=resolved_model,
            runtime_state=runtime_state,
        )

        steps: list[LoopStep | AgentStep | ModelStep | LogicStep] = [
            LoopStep(
                step_id="debate_rounds",
                steps=(
                    DelegateBatchStep(
                        step_id="debate_affirmative_batch",
                        calls_builder=callbacks.build_affirmative_calls,
                        fail_fast=True,
                    ),
                    DelegateBatchStep(
                        step_id="debate_negative_batch",
                        dependencies=("debate_affirmative_batch",),
                        calls_builder=callbacks.build_negative_calls,
                        fail_fast=True,
                    ),
                    LogicStep(
                        step_id="debate_round",
                        dependencies=("debate_affirmative_batch", "debate_negative_batch"),
                        handler=callbacks.build_round_state,
                    ),
                ),
                max_iterations=self._max_rounds,
                initial_state={
                    "rounds": [],
                    "prior_affirmative_argument": "(none)",
                    "prior_negative_argument": "(none)",
                    "should_continue": True,
                    "failure_reason": None,
                    "failure_error": None,
                },
                continue_predicate=callbacks.continue_predicate,
                state_reducer=callbacks.state_reducer,
                execution_mode="sequential",
                failure_policy="propagate_failed_state",
            )
        ]

        if self._judge_delegate is None:
            steps.extend(
                [
                    ModelStep(
                        step_id="debate_judge_model",
                        dependencies=("debate_rounds",),
                        llm_client=self._llm_client,
                        request_builder=callbacks.build_judge_request,
                        response_parser=callbacks.parse_judge_response,
                    ),
                    LogicStep(
                        step_id="debate_judge",
                        dependencies=("debate_rounds", "debate_judge_model"),
                        handler=callbacks.build_judge_output_from_model,
                    ),
                ]
            )
        else:
            steps.extend(
                [
                    AgentStep(
                        step_id="debate_judge_delegate",
                        dependencies=("debate_rounds",),
                        delegate=self._judge_delegate,
                        prompt_builder=callbacks.build_judge_prompt_from_context,
                    ),
                    LogicStep(
                        step_id="debate_judge",
                        dependencies=("debate_rounds", "debate_judge_delegate"),
                        handler=callbacks.build_judge_output_from_delegate,
                    ),
                ]
            )

        workflow = Workflow(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
            input_schema={"type": "object"},
            steps=steps,
        )
        self.workflow = workflow
        workflow_result = workflow.run(
            {},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:debate_workflow",
            dependencies=dependencies,
        )
        return _build_debate_result(
            workflow_result=workflow_result,
            runtime_state=runtime_state,
            request_id=request_id,
            dependencies=dependencies,
        )


def _build_debate_result(
    *,
    workflow_result: ExecutionResult,
    runtime_state: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> ExecutionResult:
    """Build final debate output from workflow result payloads."""
    round_state = _extract_debate_round_state(workflow_result)
    round_failure_reason = _normalize_optional_text(round_state.get("failure_reason"))
    round_failure_error = _normalize_optional_text(round_state.get("failure_error"))
    raw_rounds = round_state.get("rounds")
    normalized_rounds = (
        [dict(round_item) for round_item in raw_rounds if isinstance(round_item, Mapping)]
        if isinstance(raw_rounds, list)
        else []
    )

    judge_step = workflow_result.step_results.get("debate_judge")
    judge_output = judge_step.output if judge_step is not None else {}
    judge_status = str(judge_output.get("status", "judge_invalid_json"))

    parsed_verdict = judge_output.get("verdict")
    normalized_verdict = dict(parsed_verdict) if isinstance(parsed_verdict, Mapping) else None
    workflow_payload = workflow_result.to_dict()
    workflow_artifacts = workflow_result.output.get("artifacts", [])
    last_model_response = runtime_state.get("last_model_response")
    model_response = last_model_response if isinstance(last_model_response, LLMResponse) else None

    if round_failure_reason is not None:
        return build_failure_result(
            error=round_failure_error or "Debate round failed.",
            model_response=model_response,
            tool_results=[],
            request_id=request_id,
            dependencies=dependencies,
            metadata={"mode": "debate_pattern", "stage": "round"},
            output={
                "terminated_reason": round_failure_reason,
                "rounds": normalized_rounds,
                "verdict": None,
                "final_output": {},
                "workflow": workflow_payload,
                "artifacts": workflow_artifacts,
            },
        )

    if judge_status != "completed" or normalized_verdict is None:
        terminated_reason = (
            judge_status if judge_status in {"judge_invalid_json", "judge_invalid_schema"} else "judge_invalid_json"
        )
        raw_error = judge_output.get("error")
        error_text = (
            str(raw_error)
            if isinstance(raw_error, str) and raw_error.strip()
            else "Debate judge did not return valid JSON output."
        )
        judge_delegate_step = workflow_result.step_results.get("debate_judge_delegate")
        if (
            error_text == "Debate judge did not return valid JSON output."
            and judge_delegate_step is not None
            and isinstance(judge_delegate_step.error, str)
            and judge_delegate_step.error.strip()
        ):
            error_text = judge_delegate_step.error
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


def _extract_debate_round_state(workflow_result: ExecutionResult) -> dict[str, object]:
    """Extract final loop state mapping for debate rounds."""
    loop_step_result = workflow_result.step_results.get("debate_rounds")
    if loop_step_result is None:
        return {}
    loop_output = loop_step_result.output
    final_state = loop_output.get("final_state")
    return dict(final_state) if isinstance(final_state, Mapping) else {}


def _resolve_round_context(
    context: Mapping[str, object],
) -> tuple[int, list[dict[str, object]], str, str]:
    """Resolve round metadata and loop state payload."""
    loop_meta = context.get("_loop")
    round_number = 1
    if isinstance(loop_meta, Mapping):
        round_number = max(1, _safe_int(loop_meta.get("iteration", 1)))

    loop_state = context.get("loop_state")
    state_mapping = loop_state if isinstance(loop_state, Mapping) else {}
    raw_rounds = state_mapping.get("rounds")
    rounds = (
        [dict(round_item) for round_item in raw_rounds if isinstance(round_item, Mapping)]
        if isinstance(raw_rounds, list)
        else []
    )
    prior_affirmative = str(state_mapping.get("prior_affirmative_argument", "(none)"))
    prior_negative = str(state_mapping.get("prior_negative_argument", "(none)"))
    return round_number, rounds, prior_affirmative, prior_negative


def _extract_rounds_from_context(context: Mapping[str, object]) -> list[dict[str, object]]:
    """Extract normalized round entries from loop dependency results."""
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return []
    rounds_step = dependency_results.get("debate_rounds")
    if not isinstance(rounds_step, Mapping):
        return []
    rounds_output = rounds_step.get("output")
    if not isinstance(rounds_output, Mapping):
        return []
    final_state = rounds_output.get("final_state")
    if not isinstance(final_state, Mapping):
        return []
    raw_rounds = final_state.get("rounds")
    if not isinstance(raw_rounds, list):
        return []
    return [dict(round_item) for round_item in raw_rounds if isinstance(round_item, Mapping)]


def _extract_dependency_output(
    context: Mapping[str, object],
    *,
    dependency_id: str,
) -> dict[str, object]:
    """Extract one dependency output mapping from workflow step context."""
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return {}
    dependency_payload = dependency_results.get(dependency_id)
    if not isinstance(dependency_payload, Mapping):
        return {}
    dependency_output = dependency_payload.get("output")
    if isinstance(dependency_output, Mapping):
        return dict(dependency_output)
    return {}


def _extract_model_response_from_model_step_output(
    output: Mapping[str, object],
) -> LLMResponse | None:
    """Deserialize model response from ``ModelStep`` output mapping."""
    model_response = output.get("model_response")
    if isinstance(model_response, LLMResponse):
        return model_response
    if isinstance(model_response, Mapping):
        try:
            return LLMResponse(**dict(model_response))
        except TypeError:
            return None
    return None


def _extract_delegate_batch_call_result(
    *,
    context: Mapping[str, object],
    dependency_step_id: str,
    call_id: str,
) -> Mapping[str, object] | None:
    """Extract one call-result mapping from a batch-step dependency output."""
    return _runtime_extract_delegate_batch_call_result_from_context(
        context=context,
        dependency_step_id=dependency_step_id,
        call_id=call_id,
    )


def _extract_call_output(call_result: Mapping[str, object] | None) -> dict[str, object]:
    """Extract normalized output payload from one batch call result."""
    return _runtime_extract_call_output(call_result)


def _extract_model_text_from_output(output: Mapping[str, object]) -> str:
    """Extract normalized model text from delegate output payload."""
    model_text = output.get("model_text")
    if isinstance(model_text, str):
        return model_text.strip()
    final_output = output.get("final_output")
    if isinstance(final_output, str):
        return final_output.strip()
    if isinstance(final_output, Mapping):
        text_value = final_output.get("message")
        if isinstance(text_value, str):
            return text_value.strip()
    return ""


def _extract_call_model_response(call_result: Mapping[str, object] | None) -> LLMResponse | None:
    """Deserialize model response from one batch call result."""
    return _runtime_extract_call_model_response(call_result)


def _is_call_success(call_result: Mapping[str, object] | None) -> bool:
    """Return whether one batch call result succeeded."""
    return _runtime_is_call_success(call_result)


def _extract_call_error(
    call_result: Mapping[str, object] | None,
    *,
    fallback_message: str,
) -> str:
    """Extract one human-readable error from a batch call result."""
    return _runtime_extract_call_error(call_result, fallback_message=fallback_message)


def _render_judge_prompt(
    *,
    prompt_template: str,
    task_prompt: str,
    rounds: list[dict[str, object]],
) -> str:
    """Render judge prompt from task prompt and normalized rounds."""
    return render_prompt_template(
        template_text=prompt_template,
        variables={
            "task_prompt": task_prompt,
            "debate_rounds_json": json.dumps(rounds, ensure_ascii=True, sort_keys=True),
        },
        field_name="judge_user_prompt_template",
    )


def _extract_delegate_verdict(output: Mapping[str, object]) -> dict[str, object] | None:
    """Extract verdict payload from delegate output mappings or model text."""
    if {"winner", "rationale", "synthesis"}.issubset(output.keys()):
        return dict(output)

    final_output = output.get("final_output")
    if isinstance(final_output, Mapping) and {"winner", "rationale", "synthesis"}.issubset(final_output.keys()):
        return dict(final_output)
    if isinstance(final_output, str):
        parsed_final = _parse_json_mapping(final_output)
        if parsed_final is not None:
            return parsed_final

    model_text = output.get("model_text")
    if isinstance(model_text, str):
        return _parse_json_mapping(model_text)
    return None


def _normalize_optional_text(value: object) -> str | None:
    """Normalize optional text value."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _safe_int(value: object) -> int:
    """Convert values to int with deterministic fallback to one."""
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


__all__ = [
    "DebatePattern",
]
