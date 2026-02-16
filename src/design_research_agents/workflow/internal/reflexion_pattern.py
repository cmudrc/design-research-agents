"""Reusable ``propose_critic`` orchestration chunk."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

from design_research_agents.agent.implementations.single_step_direct_llm_agent import (
    SingleStepDirectLLMAgent,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents.agent.internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
)
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import LogicStep, LoopStep, WorkflowResult
from design_research_agents.schemas import SchemaValidationError, validate_payload_against_schema
from design_research_agents.tracing import (
    Tracer,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)
from design_research_agents.workflow.implementations.workflow_runtime import WorkflowRuntime
from design_research_agents.workflow.internal import (
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_pattern_failure_result,
    merge_dependencies,
    normalize_mapping,
    normalize_mapping_records,
    normalize_request_id_prefix,
    parse_loop_iteration,
    render_prompt_template,
    resolve_prompt_override,
    resolve_request_id_with_prefix,
)

_CRITIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["approved", "feedback", "revision_goals"],
    "properties": {
        "approved": {"type": "boolean"},
        "feedback": {"type": "string"},
        "revision_goals": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_DEFAULT_PROPOSE_CRITIC_PROPOSER_SYSTEM_PROMPT = (
    "You are a proposer. Produce a concrete draft response for the task."
)
_DEFAULT_PROPOSE_CRITIC_PROPOSER_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Iteration: $iteration",
        "Prior feedback: $prior_feedback",
        "Revision goals: $revision_goals_json",
        "Return only the revised proposal text.",
    ]
)
_DEFAULT_PROPOSE_CRITIC_CRITIC_SYSTEM_PROMPT = (
    "You are a strict critic. Return JSON only with approved, feedback, revision_goals."
)
_DEFAULT_PROPOSE_CRITIC_CRITIC_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task:",
        "$task_prompt",
        "",
        "Proposal:",
        "$proposal",
        "",
        "Critique and return structured JSON.",
    ]
)


class ReflexionPattern(Agent):
    """Propose/critique revision pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        controls: RuntimeControls | None = None,
        propose_critic_proposer_system_prompt: str | None = None,
        propose_critic_proposer_user_prompt_template: str | None = None,
        propose_critic_critic_system_prompt: str | None = None,
        propose_critic_critic_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize workflow-native orchestration settings."""
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._controls = controls or RuntimeControls()
        self._tracer = tracer
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._proposer_system_prompt = resolve_prompt_override(
            override=propose_critic_proposer_system_prompt,
            default_value=_DEFAULT_PROPOSE_CRITIC_PROPOSER_SYSTEM_PROMPT,
            field_name="propose_critic_proposer_system_prompt",
        )
        self._proposer_user_prompt_template = resolve_prompt_override(
            override=propose_critic_proposer_user_prompt_template,
            default_value=_DEFAULT_PROPOSE_CRITIC_PROPOSER_USER_PROMPT_TEMPLATE,
            field_name="propose_critic_proposer_user_prompt_template",
        )
        self._critic_system_prompt = resolve_prompt_override(
            override=propose_critic_critic_system_prompt,
            default_value=_DEFAULT_PROPOSE_CRITIC_CRITIC_SYSTEM_PROMPT,
            field_name="propose_critic_critic_system_prompt",
        )
        self._critic_user_prompt_template = resolve_prompt_override(
            override=propose_critic_critic_user_prompt_template,
            default_value=_DEFAULT_PROPOSE_CRITIC_CRITIC_USER_PROMPT_TEMPLATE,
            field_name="propose_critic_critic_user_prompt_template",
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one propose-and-critique orchestration run."""
        configured_request_id = resolve_request_id_with_prefix(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        resolved_request_id = resolve_request_id(configured_request_id)
        resolved_dependencies = normalize_dependencies(
            merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            )
        )
        normalized_input = normalize_input_payload(prompt)
        resolved_prompt = _extract_prompt(normalized_input)
        trace_scope = start_trace_run(
            agent_name="ReflexionPattern",
            request_id=resolved_request_id,
            input_payload={"prompt": resolved_prompt, "mode": "propose_critic"},
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )

        try:
            result = self._run_propose_critic(
                prompt=resolved_prompt,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute one run and emit wrapper-style stream events."""
        runtime_result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        if self._controls.streaming_enabled:
            delta_text = (
                runtime_result.model_response.text
                if runtime_result.model_response is not None
                else ""
            )
            yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=runtime_result)

    def _run_propose_critic(  # noqa: C901
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> AgentResult:
        budget_tracker = WorkflowBudgetTracker()
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        proposer = SingleStepDirectLLMAgent(
            llm_client=self._llm_client,
            system_prompt=self._proposer_system_prompt,
            tracer=self._tracer,
        )
        workflow_runtime = WorkflowRuntime(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
        )

        last_model_response = None

        def _continue_predicate(iteration: int, state: Mapping[str, object]) -> bool:
            del iteration
            if bool(state.get("approved")):
                return False
            failure_reason = state.get("failure_reason")
            return not (isinstance(failure_reason, str) and failure_reason)

        def _run_iteration(context: Mapping[str, object]) -> Mapping[str, object]:
            nonlocal last_model_response

            loop_metadata = context.get("_loop")
            if not isinstance(loop_metadata, Mapping):
                raise ValueError("Loop metadata is required for propose_critic iteration.")

            raw_iteration = loop_metadata.get("iteration")
            iteration = parse_loop_iteration(
                raw_iteration,
                error_prefix="Propose and critique loop iteration",
            )

            loop_state = context.get("loop_state")
            state_mapping = loop_state if isinstance(loop_state, Mapping) else {}
            current_feedback = str(state_mapping.get("feedback", ""))
            revision_goals_raw = state_mapping.get("revision_goals")
            current_goals = (
                [str(goal) for goal in revision_goals_raw]
                if isinstance(revision_goals_raw, list)
                else []
            )

            propose_prompt = render_prompt_template(
                template_text=self._proposer_user_prompt_template,
                variables={
                    "task_prompt": prompt,
                    "iteration": iteration,
                    "prior_feedback": current_feedback or "(none)",
                    "revision_goals_json": json.dumps(current_goals, sort_keys=True),
                },
                field_name="propose_critic_proposer_user_prompt_template",
            )
            propose_result = proposer.run(
                propose_prompt,
                request_id=f"{request_id}:propose-{iteration}",
                dependencies=dependencies,
            )
            if propose_result.model_response is not None:
                last_model_response = propose_result.model_response
                budget_tracker.add_model_response(propose_result.model_response)
            current_proposal = str(propose_result.output.get("model_text", "")).strip()

            critic_messages = [
                LLMMessage(role="system", content=self._critic_system_prompt),
                LLMMessage(
                    role="user",
                    content=render_prompt_template(
                        template_text=self._critic_user_prompt_template,
                        variables={"task_prompt": prompt, "proposal": current_proposal},
                        field_name="propose_critic_critic_user_prompt_template",
                    ),
                ),
            ]
            critic_params = LLMChatParams(
                response_schema=dict(_CRITIC_SCHEMA),
                provider_options={
                    "agent": "ReflexionPattern",
                    "mode": "propose_critic",
                    "phase": "critic",
                },
            )
            critic_span_id = start_model_call(
                model=resolved_model,
                messages=critic_messages,
                params=critic_params,
                metadata={
                    "agent": "ReflexionPattern",
                    "mode": "propose_critic",
                    "phase": "critic",
                },
            )
            try:
                critic_response = self._llm_client.chat(
                    critic_messages,
                    model=resolved_model,
                    params=critic_params,
                )
            except Exception as exc:
                finish_model_call(critic_span_id, error=str(exc), model=resolved_model)
                raise
            finish_model_call(critic_span_id, response=critic_response)
            last_model_response = critic_response
            budget_tracker.add_model_response(critic_response)

            parsed_critique = _parse_json_mapping(critic_response.text)
            if parsed_critique is None:
                return {
                    "failure_reason": "critic_invalid_json",
                    "failure_error": "Critic did not return valid JSON output.",
                    "proposal": current_proposal,
                }

            try:
                validate_payload_against_schema(
                    payload=parsed_critique,
                    schema=_CRITIC_SCHEMA,
                    location="propose_critic.critic",
                )
            except SchemaValidationError as exc:
                return {
                    "failure_reason": "critic_invalid_schema",
                    "failure_error": f"Critic output failed schema validation: {exc}",
                    "proposal": current_proposal,
                }

            revision_goals_raw = parsed_critique.get("revision_goals")
            revision_goals = (
                [str(goal) for goal in revision_goals_raw]
                if isinstance(revision_goals_raw, list)
                else []
            )
            return {
                "failure_reason": None,
                "failure_error": None,
                "proposal": current_proposal,
                "approved": bool(parsed_critique.get("approved")),
                "feedback": str(parsed_critique.get("feedback", "")),
                "revision_goals": revision_goals,
            }

        def _state_reducer(
            state: Mapping[str, object],
            iteration_result: WorkflowResult,
            iteration: int,
        ) -> Mapping[str, object]:
            next_state = dict(state)
            critique_iterations = normalize_mapping_records(next_state.get("critique_iterations"))
            next_state["critique_iterations"] = critique_iterations

            step_result = iteration_result.step_results.get("propose_critic_iteration")
            if step_result is None:
                next_state["failure_reason"] = "iteration_failed"
                next_state["failure_error"] = (
                    "Iteration result missing propose_critic_iteration output."
                )
                return next_state

            if not step_result.success:
                next_state["failure_reason"] = "iteration_failed"
                next_state["failure_error"] = step_result.error or "Workflow iteration failed."
                return next_state

            iteration_output = step_result.output
            maybe_failure_reason = iteration_output.get("failure_reason")
            if isinstance(maybe_failure_reason, str) and maybe_failure_reason:
                maybe_failure_error = iteration_output.get("failure_error")
                next_state["failure_reason"] = maybe_failure_reason
                next_state["failure_error"] = (
                    str(maybe_failure_error)
                    if maybe_failure_error is not None
                    else "Critic iteration failed."
                )
                next_state["proposal"] = str(
                    iteration_output.get("proposal", next_state.get("proposal", ""))
                )
                return next_state

            proposal = str(iteration_output.get("proposal", next_state.get("proposal", "")))
            approved = bool(iteration_output.get("approved"))
            feedback = str(iteration_output.get("feedback", ""))
            revision_goals_raw = iteration_output.get("revision_goals")
            revision_goals = (
                [str(goal) for goal in revision_goals_raw]
                if isinstance(revision_goals_raw, list)
                else []
            )
            critique_iterations.append(
                {
                    "iteration": iteration,
                    "proposal": proposal,
                    "approved": approved,
                    "feedback": feedback,
                    "revision_goals": revision_goals,
                }
            )

            next_state["proposal"] = proposal
            next_state["approved"] = approved
            next_state["feedback"] = feedback
            next_state["revision_goals"] = revision_goals
            next_state["failure_reason"] = None
            next_state["failure_error"] = None
            return next_state

        workflow_result = workflow_runtime.run(
            steps=[
                LoopStep(
                    step_id="propose_critic_loop",
                    steps=(
                        LogicStep(
                            step_id="propose_critic_iteration",
                            handler=_run_iteration,
                        ),
                    ),
                    max_iterations=self._controls.max_iterations,
                    initial_state={
                        "proposal": "",
                        "approved": False,
                        "feedback": "",
                        "revision_goals": [],
                        "failure_reason": None,
                        "failure_error": None,
                        "critique_iterations": [],
                    },
                    continue_predicate=_continue_predicate,
                    state_reducer=_state_reducer,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                )
            ],
            context={"prompt": prompt},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:propose_critic_loop",
            dependencies=dependencies,
        )
        loop_step_result = workflow_result.step_results.get("propose_critic_loop")
        if loop_step_result is None:
            raise RuntimeError("Propose and critique loop step result is missing.")
        loop_output = loop_step_result.output
        final_state = normalize_mapping(loop_output.get("final_state"))
        critique_iterations = normalize_mapping_records(final_state.get("critique_iterations"))
        current_proposal = str(final_state.get("proposal", ""))
        approved = bool(final_state.get("approved"))
        failure_reason_raw = final_state.get("failure_reason")
        failure_reason = (
            str(failure_reason_raw)
            if isinstance(failure_reason_raw, str) and failure_reason_raw
            else None
        )
        failure_error_raw = final_state.get("failure_error")
        failure_error = (
            str(failure_error_raw)
            if isinstance(failure_error_raw, str) and failure_error_raw
            else None
        )

        loop_terminated_reason = str(loop_output.get("terminated_reason", "max_iterations_reached"))
        if approved:
            terminated_reason = "approved"
        elif failure_reason is not None:
            terminated_reason = failure_reason
        else:
            terminated_reason = "max_iterations_reached"

        if loop_terminated_reason == "iteration_failed" or failure_reason == "iteration_failed":
            error_message = failure_error or "Workflow loop iteration failed."
            raise RuntimeError(error_message)

        if failure_reason in {"critic_invalid_json", "critic_invalid_schema"}:
            failure = build_pattern_failure_result(
                error=failure_error or "Critic iteration failed.",
                model_response=last_model_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={"stage": "critic", "mode": "propose_critic"},
                output={
                    "proposal": current_proposal,
                    "critique_iterations": critique_iterations,
                    "terminated_reason": failure_reason,
                },
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode="propose_critic",
                resolved_mode="propose_critic",
                controls=self._controls,
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        result = AgentResult(
            output={
                "proposal": current_proposal,
                "critique_iterations": critique_iterations,
                "terminated_reason": terminated_reason,
                "approved": approved,
            },
            success=approved,
            tool_results=[],
            model_response=last_model_response,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "mode": "propose_critic",
                "iterations": len(critique_iterations),
            },
        )
        return attach_runtime_metadata(
            agent_result=result,
            requested_mode="propose_critic",
            resolved_mode="propose_critic",
            controls=self._controls,
            budget_metadata=budget_tracker.as_metadata(controls=self._controls),
            extra_metadata={
                "loop": {
                    "iterations": loop_output.get("iterations", self._controls.max_iterations),
                    "iterations_executed": loop_output.get("iterations_executed", 0),
                    "terminated_reason": loop_terminated_reason,
                }
            },
        )


__all__ = [
    "ReflexionPattern",
]
