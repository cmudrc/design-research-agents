"""Shared constants and callbacks for propose_critic/propose-critic patterns."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._llm import LLMMessage, LLMRequest, LLMResponse
from design_research_agents._implementations._shared._agent_internal._input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents._runtime._patterns import (
    MODE_PROPOSE_CRITIC,
    WorkflowBudgetTracker,
    normalize_mapping_records,
    parse_loop_iteration,
    render_prompt_template,
)
from design_research_agents._schemas import (
    SchemaValidationError,
    validate_payload_against_schema,
)
from design_research_agents._skills import SkillsContext, inject_skills_into_prompt_pair

CRITIC_SCHEMA: dict[str, object] = {
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
        "reasoning": {"type": "string"},
    },
}

DEFAULT_PROPOSER_SYSTEM_PROMPT = "You are a proposer. Produce a concrete draft response for the task."
DEFAULT_PROPOSER_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Iteration: $iteration",
        "Prior feedback: $prior_feedback",
        "Revision goals: $revision_goals_json",
        "Return only the revised proposal text.",
    ]
)
DEFAULT_CRITIC_SYSTEM_PROMPT = (
    "You are a strict critic. Return JSON only with approved, feedback, revision_goals, "
    "and reasoning. The reasoning field should explain, in your own words, why you reached "
    "this verdict, separate from the feedback given to the proposer."
)
DEFAULT_CRITIC_USER_PROMPT_TEMPLATE = "\n".join(
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


class ProposeCriticLoopCallbacks:
    """Callback bundle for propose_critic loop prompt and state handling."""

    def __init__(
        self,
        *,
        resolved_model: str,
        task_prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        proposer_user_prompt_template: str,
        critic_system_prompt: str,
        critic_user_prompt_template: str,
        budget_tracker: WorkflowBudgetTracker,
        skills_context: SkillsContext | None = None,
    ) -> None:
        """Store loop dependencies shared by callback methods."""
        self.resolved_model = resolved_model
        self.task_prompt = task_prompt
        self.request_id = request_id
        self.dependencies = dependencies
        self.proposer_user_prompt_template = proposer_user_prompt_template
        self.critic_system_prompt = critic_system_prompt
        self.critic_user_prompt_template = critic_user_prompt_template
        self.budget_tracker = budget_tracker
        self.skills_context = skills_context
        self.last_model_response: LLMResponse | None = None

    def continue_predicate(self, iteration: int, state: Mapping[str, object]) -> bool:
        """Continue until proposal is approved or an unrecoverable failure occurs."""
        del iteration
        if bool(state.get("approved")):
            return False
        failure_reason = state.get("failure_reason")
        return not (isinstance(failure_reason, str) and failure_reason)

    def build_proposer_prompt(self, context: Mapping[str, object]) -> str:
        """Build proposer prompt for one loop iteration."""
        iteration, current_feedback, current_goals = self._extract_iteration_state(context)
        return render_prompt_template(
            template_text=self.proposer_user_prompt_template,
            variables={
                "task_prompt": self.task_prompt,
                "iteration": iteration,
                "prior_feedback": current_feedback or "(none)",
                "revision_goals_json": json.dumps(current_goals, sort_keys=True),
            },
            field_name="proposer_user_prompt_template",
        )

    def build_critic_prompt(self, context: Mapping[str, object]) -> str:
        """Build critic prompt from the latest proposer output in loop context."""
        proposal = _extract_proposer_text(context)
        return render_prompt_template(
            template_text=self.critic_user_prompt_template,
            variables={"task_prompt": self.task_prompt, "proposal": proposal},
            field_name="critic_user_prompt_template",
        )

    def build_critic_request(self, context: Mapping[str, object]) -> LLMRequest:
        """Build model request payload for direct critic invocation."""
        critic_prompt = self.build_critic_prompt(context)
        metadata: dict[str, object] = {
            "agent": "ProposeCriticPattern",
            "mode": MODE_PROPOSE_CRITIC,
            "phase": "critic",
            "request_id": self.request_id,
        }
        system_prompt, user_prompt = inject_skills_into_prompt_pair(
            system_prompt=self.critic_system_prompt,
            user_prompt=critic_prompt,
            skills_context=self.skills_context,
            include_catalog=False,
        )
        return LLMRequest(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ],
            model=self.resolved_model,
            response_schema=dict(CRITIC_SCHEMA),
            metadata=dict(metadata),
            provider_options=dict(metadata),
        )

    def parse_critic_model_response(
        self,
        response: LLMResponse,
        context: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Parse model critic response into structured critique payload."""
        del context
        return {"critique": _parse_json_mapping(response.text)}

    def build_iteration_from_model(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Build one iteration state from proposer and direct model-critic steps."""
        proposal = _extract_proposer_text(context)
        proposer_step = _extract_dependency_output(
            context=context,
            dependency_id="propose_critic_proposer",
        )
        self._record_step_model_response(proposer_step)

        critic_step = _extract_dependency_output(
            context=context,
            dependency_id="propose_critic_critic_model",
        )
        self._record_step_model_response(critic_step)
        critic_step_success = critic_step.get("success") is True
        if not critic_step_success:
            error_text = str(critic_step.get("error", "Critic model step failed."))
            return {
                "failure_reason": "iteration_failed",
                "failure_error": error_text,
                "proposal": proposal,
            }

        critic_step_output = critic_step.get("output")
        normalized_step_output = dict(critic_step_output) if isinstance(critic_step_output, Mapping) else {}
        parsed_payload = normalized_step_output.get("parsed")
        parsed_mapping = dict(parsed_payload) if isinstance(parsed_payload, Mapping) else {}
        critique_payload = parsed_mapping.get("critique")
        return self._build_critique_result(
            proposal=proposal,
            parsed_critique=critique_payload,
        )

    def build_iteration_from_delegate(
        self,
        context: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Build one iteration state from proposer and delegate-critic steps."""
        proposal = _extract_proposer_text(context)
        proposer_step = _extract_dependency_output(
            context=context,
            dependency_id="propose_critic_proposer",
        )
        self._record_step_model_response(proposer_step)

        critic_step = _extract_dependency_output(
            context=context,
            dependency_id="propose_critic_critic_delegate",
        )
        self._record_step_model_response(critic_step)
        critic_delegate_output = _extract_step_delegate_output(critic_step)
        critic_step_success = critic_step.get("success") is True
        if not critic_step_success and not critic_delegate_output:
            error_text = str(critic_step.get("error", "Critic delegate step failed."))
            return {
                "failure_reason": "iteration_failed",
                "failure_error": error_text,
                "proposal": proposal,
            }

        critic_text = _extract_delegate_text(critic_delegate_output)
        parsed_critique = _parse_json_mapping(critic_text)
        return self._build_critique_result(
            proposal=proposal,
            parsed_critique=parsed_critique,
        )

    def state_reducer(
        self,
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        """Fold one iteration result into accumulated propose_critic loop state."""
        next_state = dict(state)
        critique_iterations = normalize_mapping_records(next_state.get("critique_iterations"))
        next_state["critique_iterations"] = critique_iterations

        step_result = iteration_result.step_results.get("propose_critic_iteration")
        if step_result is None:
            next_state["failure_reason"] = "iteration_failed"
            next_state["failure_error"] = "Iteration result missing propose_critic_iteration output."
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
                str(maybe_failure_error) if maybe_failure_error is not None else "Critic iteration failed."
            )
            next_state["proposal"] = str(iteration_output.get("proposal", next_state.get("proposal", "")))
            return next_state

        proposal = str(iteration_output.get("proposal", next_state.get("proposal", "")))
        approved = bool(iteration_output.get("approved"))
        feedback = str(iteration_output.get("feedback", ""))
        revision_goals_raw = iteration_output.get("revision_goals")
        revision_goals = [str(goal) for goal in revision_goals_raw] if isinstance(revision_goals_raw, list) else []
        reasoning_raw = iteration_output.get("reasoning")
        reasoning = str(reasoning_raw) if isinstance(reasoning_raw, str) else ""
        critique_iterations.append(
            {
                "iteration": iteration,
                "proposal": proposal,
                "approved": approved,
                "feedback": feedback,
                "revision_goals": revision_goals,
                "reasoning": reasoning,
            }
        )

        next_state["proposal"] = proposal
        next_state["approved"] = approved
        next_state["feedback"] = feedback
        next_state["revision_goals"] = revision_goals
        next_state["reasoning"] = reasoning
        next_state["failure_reason"] = None
        next_state["failure_error"] = None
        return next_state

    def _extract_iteration_state(
        self,
        context: Mapping[str, object],
    ) -> tuple[int, str, list[str]]:
        """Extract iteration index and prior critique state from loop context."""
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
        current_goals = [str(goal) for goal in revision_goals_raw] if isinstance(revision_goals_raw, list) else []
        return iteration, current_feedback, current_goals

    def _record_step_model_response(
        self,
        step_payload: Mapping[str, object],
    ) -> None:
        """Record model response metrics from one serialized dependency step payload."""
        model_response = _extract_step_model_response(step_payload)
        if model_response is None:
            return
        self.last_model_response = model_response
        self.budget_tracker.add_model_response(model_response)

    def _build_critique_result(
        self,
        *,
        proposal: str,
        parsed_critique: object,
    ) -> Mapping[str, object]:
        """Validate parsed critique payload and normalize iteration output."""
        if not isinstance(parsed_critique, Mapping):
            return {
                "failure_reason": "critic_invalid_json",
                "failure_error": "Critic did not return valid JSON output.",
                "proposal": proposal,
            }

        normalized_critique = dict(parsed_critique)
        try:
            validate_payload_against_schema(
                payload=normalized_critique,
                schema=CRITIC_SCHEMA,
                location="propose_critic.critic",
            )
        except SchemaValidationError as exc:
            return {
                "failure_reason": "critic_invalid_schema",
                "failure_error": f"Critic output failed schema validation: {exc}",
                "proposal": proposal,
            }

        revision_goals_raw = normalized_critique.get("revision_goals")
        revision_goals = [str(goal) for goal in revision_goals_raw] if isinstance(revision_goals_raw, list) else []
        reasoning_raw = normalized_critique.get("reasoning")
        reasoning = str(reasoning_raw) if isinstance(reasoning_raw, str) else ""
        return {
            "failure_reason": None,
            "failure_error": None,
            "proposal": proposal,
            "approved": bool(normalized_critique.get("approved")),
            "feedback": str(normalized_critique.get("feedback", "")),
            "revision_goals": revision_goals,
            "reasoning": reasoning,
        }


def _extract_dependency_output(
    *,
    context: Mapping[str, object],
    dependency_id: str,
) -> Mapping[str, object]:
    """Extract one dependency step payload from workflow step context."""
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return {}
    dependency_payload = dependency_results.get(dependency_id)
    if not isinstance(dependency_payload, Mapping):
        return {}
    return dict(dependency_payload)


def _extract_step_delegate_output(step_payload: Mapping[str, object]) -> Mapping[str, object]:
    """Extract serialized delegate output payload from one step-result mapping."""
    raw_step_output = step_payload.get("output")
    if not isinstance(raw_step_output, Mapping):
        return {}
    delegate_output = raw_step_output.get("output")
    if not isinstance(delegate_output, Mapping):
        return {}
    return dict(delegate_output)


def _extract_step_model_response(step_payload: Mapping[str, object]) -> LLMResponse | None:
    """Extract serialized ``LLMResponse`` from one step-result mapping."""
    raw_step_output = step_payload.get("output")
    if not isinstance(raw_step_output, Mapping):
        return None
    raw_model_response = raw_step_output.get("model_response")
    if not isinstance(raw_model_response, Mapping):
        return None
    try:
        return LLMResponse(**dict(raw_model_response))
    except TypeError:
        return None


def _extract_proposer_text(context: Mapping[str, object]) -> str:
    """Extract proposer text payload from loop dependency context."""
    proposer_step = _extract_dependency_output(
        context=context,
        dependency_id="propose_critic_proposer",
    )
    proposer_output = _extract_step_delegate_output(proposer_step)
    return _extract_delegate_text(proposer_output).strip()


def _extract_delegate_text(output: Mapping[str, object]) -> str:
    """Extract best-effort text payload from serialized delegate output mapping."""
    final_output = output.get("final_output")
    if isinstance(final_output, str):
        return final_output
    if isinstance(final_output, Mapping):
        return json.dumps(dict(final_output), ensure_ascii=True, sort_keys=True)
    model_text = output.get("model_text")
    if isinstance(model_text, str):
        return model_text
    return json.dumps(dict(output), ensure_ascii=True, sort_keys=True)


__all__ = [
    "CRITIC_SCHEMA",
    "DEFAULT_CRITIC_SYSTEM_PROMPT",
    "DEFAULT_CRITIC_USER_PROMPT_TEMPLATE",
    "DEFAULT_PROPOSER_SYSTEM_PROMPT",
    "DEFAULT_PROPOSER_USER_PROMPT_TEMPLATE",
    "ProposeCriticLoopCallbacks",
]
