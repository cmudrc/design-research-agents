"""Shared constants for reflexion/propose-critic workflow patterns."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol

from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.implementations.shared.agent_internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.implementations.shared.workflow_internal import (
    WorkflowBudgetTracker,
    normalize_mapping_records,
    parse_loop_iteration,
    render_prompt_template,
)
from design_research_agents.schemas import (
    SchemaValidationError,
    validate_payload_against_schema,
)
from design_research_agents.tracing import finish_model_call, start_model_call

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
    },
}

DEFAULT_PROPOSER_SYSTEM_PROMPT = (
    "You are a proposer. Produce a concrete draft response for the task."
)
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
    "You are a strict critic. Return JSON only with approved, feedback, revision_goals."
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


class ReflexionLoopCallbacks:
    """Callback bundle used by reflexion propose/critic loop runtime."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        proposer_agent: ProposerAgent,
        resolved_model: str,
        task_prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        proposer_user_prompt_template: str,
        critic_system_prompt: str,
        critic_user_prompt_template: str,
        budget_tracker: WorkflowBudgetTracker,
    ) -> None:
        """Store loop dependencies shared by callback methods.

        Args:
            llm_client: LLM client used for critic calls.
            proposer_agent: Proposer agent dependency exposing ``run``.
            resolved_model: Resolved model id for critic calls.
            task_prompt: User task prompt text.
            request_id: Resolved request id.
            dependencies: Normalized dependency mapping.
            proposer_user_prompt_template: Proposer user prompt template.
            critic_system_prompt: Critic system prompt.
            critic_user_prompt_template: Critic user prompt template.
            budget_tracker: Budget tracker collecting model metrics.
        """
        self.llm_client = llm_client
        self.proposer_agent = proposer_agent
        self.resolved_model = resolved_model
        self.task_prompt = task_prompt
        self.request_id = request_id
        self.dependencies = dependencies
        self.proposer_user_prompt_template = proposer_user_prompt_template
        self.critic_system_prompt = critic_system_prompt
        self.critic_user_prompt_template = critic_user_prompt_template
        self.budget_tracker = budget_tracker
        self.last_model_response: LLMResponse | None = None

    def continue_predicate(self, iteration: int, state: Mapping[str, object]) -> bool:
        """Continue until proposal is approved or an unrecoverable failure occurs.

        Args:
            iteration: One-based loop iteration index.
            state: Current loop state before this iteration.

        Returns:
            ``True`` when another propose/critic iteration should run.
        """
        del iteration
        if bool(state.get("approved")):
            return False
        failure_reason = state.get("failure_reason")
        return not (isinstance(failure_reason, str) and failure_reason)

    def run_iteration(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute one propose-then-critic iteration.

        Args:
            context: Step context containing loop metadata and loop state.

        Returns:
            Iteration payload containing proposal, approval, and feedback fields.

        Raises:
            ValueError: If required loop metadata is missing.
        """
        iteration, current_feedback, current_goals = self._extract_iteration_state(context)
        current_proposal = self._run_proposer(iteration, current_feedback, current_goals)
        critic_response = self._run_critic(current_proposal)

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
                schema=CRITIC_SCHEMA,
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

    def _extract_iteration_state(
        self,
        context: Mapping[str, object],
    ) -> tuple[int, str, list[str]]:
        """Extract iteration index and prior critique state from loop context.

        Args:
            context: Loop step context containing ``_loop`` metadata and loop state.

        Returns:
            Tuple of iteration number, prior feedback text, and revision goals.

        Raises:
            ValueError: If required loop metadata is missing.
        """
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
        return iteration, current_feedback, current_goals

    def _run_proposer(
        self,
        iteration: int,
        current_feedback: str,
        current_goals: list[str],
    ) -> str:
        """Run proposer agent and update budget/model-response state.

        Args:
            iteration: One-based iteration number.
            current_feedback: Feedback text from the prior iteration.
            current_goals: Revision goals from the prior iteration.

        Returns:
            Normalized proposal text emitted by the proposer.
        """
        propose_prompt = render_prompt_template(
            template_text=self.proposer_user_prompt_template,
            variables={
                "task_prompt": self.task_prompt,
                "iteration": iteration,
                "prior_feedback": current_feedback or "(none)",
                "revision_goals_json": json.dumps(current_goals, sort_keys=True),
            },
            field_name="propose_critic_proposer_user_prompt_template",
        )
        propose_result = self.proposer_agent.run(
            propose_prompt,
            request_id=f"{self.request_id}:propose-{iteration}",
            dependencies=self.dependencies,
        )
        if propose_result.model_response is not None:
            self.last_model_response = propose_result.model_response
            self.budget_tracker.add_model_response(propose_result.model_response)
        return str(propose_result.output.get("model_text", "")).strip()

    def _run_critic(self, proposal: str) -> LLMResponse:
        """Run critic model call for one proposal revision.

        Args:
            proposal: Proposed draft text to critique.

        Returns:
            Critic model response payload.

        Raises:
            Exception: Propagates model client failures.
        """
        critic_messages = [
            LLMMessage(role="system", content=self.critic_system_prompt),
            LLMMessage(
                role="user",
                content=render_prompt_template(
                    template_text=self.critic_user_prompt_template,
                    variables={"task_prompt": self.task_prompt, "proposal": proposal},
                    field_name="propose_critic_critic_user_prompt_template",
                ),
            ),
        ]
        critic_params = LLMChatParams(
            response_schema=dict(CRITIC_SCHEMA),
            provider_options={
                "agent": "ReflexionPattern",
                "mode": "propose_critic",
                "phase": "critic",
            },
        )
        critic_span_id = start_model_call(
            model=self.resolved_model,
            messages=critic_messages,
            params=critic_params,
            metadata={
                "agent": "ReflexionPattern",
                "mode": "propose_critic",
                "phase": "critic",
            },
        )
        try:
            critic_response = self.llm_client.chat(
                critic_messages,
                model=self.resolved_model,
                params=critic_params,
            )
        except Exception as exc:
            finish_model_call(critic_span_id, error=str(exc), model=self.resolved_model)
            raise
        finish_model_call(critic_span_id, response=critic_response)
        self.last_model_response = critic_response
        self.budget_tracker.add_model_response(critic_response)
        return critic_response

    def state_reducer(
        self,
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        """Fold one iteration result into accumulated reflexion loop state.

        Args:
            state: Current aggregate loop state.
            iteration_result: Workflow result produced by this iteration body.
            iteration: One-based iteration index.

        Returns:
            Updated state with proposal, feedback, and critique history.
        """
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


class ProposerAgent(Protocol):
    """Protocol for proposer delegates used by reflexion loop callbacks."""

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one proposer step and return an execution result.

        Args:
            prompt: Prompt text for the proposer run.
            request_id: Optional request identifier.
            dependencies: Optional dependency payload mapping.

        Returns:
            Execution result emitted by the proposer delegate.
        """


__all__ = [
    "CRITIC_SCHEMA",
    "DEFAULT_CRITIC_SYSTEM_PROMPT",
    "DEFAULT_CRITIC_USER_PROMPT_TEMPLATE",
    "DEFAULT_PROPOSER_SYSTEM_PROMPT",
    "DEFAULT_PROPOSER_USER_PROMPT_TEMPLATE",
    "ReflexionLoopCallbacks",
]
