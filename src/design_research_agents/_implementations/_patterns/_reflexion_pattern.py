"""Reusable ``propose_critic`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents._contracts._agent import Agent, ExecutionResult
from design_research_agents._contracts._llm import LLMClient
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._contracts._workflow import (
    AgentStep,
    LogicStep,
    LoopStep,
    ModelStep,
    WorkflowDelegate,
    WorkflowStep,
)
from design_research_agents._implementations._agents._direct_llm_call import (
    DirectLLMCall,
)
from design_research_agents._implementations._shared._agent_internal._input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents._implementations._shared._agent_internal._model_resolution import (
    resolve_agent_model,
)
from design_research_agents._implementations._shared._agent_internal._run_options import (
    normalize_input_payload,
)
from design_research_agents._implementations._shared._workflow_internal._reflexion_helpers import (
    DEFAULT_CRITIC_SYSTEM_PROMPT,
    DEFAULT_CRITIC_USER_PROMPT_TEMPLATE,
    DEFAULT_PROPOSER_SYSTEM_PROMPT,
    DEFAULT_PROPOSER_USER_PROMPT_TEMPLATE,
    ReflexionLoopCallbacks,
)
from design_research_agents._runtime._patterns import (
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_pattern_failure_result,
    execute_pattern_with_trace,
    normalize_mapping,
    normalize_mapping_records,
    normalize_request_id_prefix,
    resolve_pattern_run_context,
    resolve_prompt_override,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow.workflow import Workflow


class ReflexionPattern(Agent):
    """Propose/critique revision pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        proposer_delegate: WorkflowDelegate | None = None,
        critic_delegate: WorkflowDelegate | None = None,
        max_iterations: int = 3,
        proposer_system_prompt: str | None = None,
        proposer_user_prompt_template: str | None = None,
        critic_system_prompt: str | None = None,
        critic_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize workflow-native orchestration settings.

        Args:
            llm_client: LLM client used by proposer and critic calls.
            tool_runtime: Tool runtime used by loop execution runtime.
            proposer_delegate: Optional proposer delegate override.
            critic_delegate: Optional critic delegate override.
            max_iterations: Maximum propose/critic iterations per run.
            proposer_system_prompt: Optional override for proposer system prompt.
            proposer_user_prompt_template: Optional proposer user prompt template.
            critic_system_prompt: Optional override for critic system prompt.
            critic_user_prompt_template: Optional critic user prompt template.
            default_request_id_prefix: Optional prefix used to derive request ids.
            default_dependencies: Dependency defaults merged into each run.
            tracer: Optional tracer used for run-level instrumentation.

        Raises:
            ValueError: If ``max_iterations`` is invalid.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._proposer_delegate = proposer_delegate
        self._critic_delegate = critic_delegate
        self._max_iterations = max_iterations
        self._tracer = tracer
        self.workflow: Workflow | None = None
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._proposer_system_prompt = resolve_prompt_override(
            override=proposer_system_prompt,
            default_value=DEFAULT_PROPOSER_SYSTEM_PROMPT,
            field_name="proposer_system_prompt",
        )
        self._proposer_user_prompt_template = resolve_prompt_override(
            override=proposer_user_prompt_template,
            default_value=DEFAULT_PROPOSER_USER_PROMPT_TEMPLATE,
            field_name="proposer_user_prompt_template",
        )
        self._critic_system_prompt = resolve_prompt_override(
            override=critic_system_prompt,
            default_value=DEFAULT_CRITIC_SYSTEM_PROMPT,
            field_name="critic_system_prompt",
        )
        self._critic_user_prompt_template = resolve_prompt_override(
            override=critic_user_prompt_template,
            default_value=DEFAULT_CRITIC_USER_PROMPT_TEMPLATE,
            field_name="critic_user_prompt_template",
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one propose-and-critique orchestration run.

        Args:
            prompt: Task prompt to iteratively refine.
            request_id: Optional request id for tracing and correlation.
            dependencies: Optional dependency overrides for this run.

        Returns:
            Pattern result containing final proposal and critique history.

        Raises:
            Exception: Propagates runtime failures from proposer/critic loop execution.
        """
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=self._default_request_id_prefix,
            default_dependencies=self._default_dependencies,
            request_id=request_id,
            dependencies=dependencies,
        )
        normalized_input = normalize_input_payload(prompt)
        resolved_prompt = _extract_prompt(normalized_input)
        return execute_pattern_with_trace(
            agent_name="ReflexionPattern",
            request_id=run_context.request_id,
            input_payload={"prompt": resolved_prompt, "mode": "propose_critic"},
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            runner=lambda: self._run_propose_critic(
                prompt=resolved_prompt,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _run_propose_critic(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Run propose/critic loop until approval or termination.

        Args:
            prompt: Task prompt to iteratively refine.
            request_id: Resolved request id for this orchestration run.
            dependencies: Normalized dependency mapping for this run.

        Returns:
            Final reflexion pattern result.

        Raises:
            RuntimeError: If loop execution fails irrecoverably.
        """
        budget_tracker = WorkflowBudgetTracker()
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        proposer = self._proposer_delegate
        if proposer is None:
            proposer = DirectLLMCall(
                llm_client=self._llm_client,
                system_prompt=self._proposer_system_prompt,
                tracer=self._tracer,
            )
        callbacks = ReflexionLoopCallbacks(
            resolved_model=resolved_model,
            task_prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            proposer_user_prompt_template=self._proposer_user_prompt_template,
            critic_system_prompt=self._critic_system_prompt,
            critic_user_prompt_template=self._critic_user_prompt_template,
            budget_tracker=budget_tracker,
        )
        loop_steps: tuple[WorkflowStep, ...]
        if self._critic_delegate is None:
            loop_steps = (
                AgentStep(
                    step_id="propose_critic_proposer",
                    delegate=proposer,
                    prompt_builder=callbacks.build_proposer_prompt,
                ),
                ModelStep(
                    step_id="propose_critic_critic_model",
                    dependencies=("propose_critic_proposer",),
                    llm_client=self._llm_client,
                    request_builder=callbacks.build_critic_request,
                    response_parser=callbacks.parse_critic_model_response,
                ),
                LogicStep(
                    step_id="propose_critic_iteration",
                    dependencies=(
                        "propose_critic_proposer",
                        "propose_critic_critic_model",
                    ),
                    handler=callbacks.build_iteration_from_model,
                ),
            )
        else:
            loop_steps = (
                AgentStep(
                    step_id="propose_critic_proposer",
                    delegate=proposer,
                    prompt_builder=callbacks.build_proposer_prompt,
                ),
                AgentStep(
                    step_id="propose_critic_critic_delegate",
                    dependencies=("propose_critic_proposer",),
                    delegate=self._critic_delegate,
                    prompt_builder=callbacks.build_critic_prompt,
                ),
                LogicStep(
                    step_id="propose_critic_iteration",
                    dependencies=(
                        "propose_critic_proposer",
                        "propose_critic_critic_delegate",
                    ),
                    handler=callbacks.build_iteration_from_delegate,
                ),
            )

        self.workflow = Workflow(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
            input_schema={"type": "object"},
            base_context={"prompt": prompt},
            steps=[
                LoopStep(
                    step_id="propose_critic_loop",
                    steps=loop_steps,
                    max_iterations=self._max_iterations,
                    initial_state={
                        "proposal": "",
                        "approved": False,
                        "feedback": "",
                        "revision_goals": [],
                        "failure_reason": None,
                        "failure_error": None,
                        "critique_iterations": [],
                    },
                    continue_predicate=callbacks.continue_predicate,
                    state_reducer=callbacks.state_reducer,
                    execution_mode="sequential",
                    failure_policy="propagate_failed_state",
                )
            ],
        )

        workflow_result = self.workflow.run(
            {},
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
        failure_reason = str(failure_reason_raw) if isinstance(failure_reason_raw, str) and failure_reason_raw else None
        failure_error_raw = final_state.get("failure_error")
        failure_error = str(failure_error_raw) if isinstance(failure_error_raw, str) and failure_error_raw else None

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
                model_response=callbacks.last_model_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={"stage": "critic", "mode": "propose_critic"},
                output={
                    "proposal": current_proposal,
                    "critique_iterations": critique_iterations,
                    "terminated_reason": failure_reason,
                    "workflow": workflow_result.to_dict(),
                    "artifacts": workflow_result.output.get("artifacts", []),
                },
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode="propose_critic",
                resolved_mode="propose_critic",
                budget_metadata=budget_tracker.as_metadata(),
                extra_metadata=None,
            )

        result = ExecutionResult(
            output={
                "proposal": current_proposal,
                "critique_iterations": critique_iterations,
                "terminated_reason": terminated_reason,
                "approved": approved,
                "workflow": workflow_result.to_dict(),
                "artifacts": workflow_result.output.get("artifacts", []),
            },
            success=approved,
            tool_results=[],
            model_response=callbacks.last_model_response,
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
            budget_metadata=budget_tracker.as_metadata(),
            extra_metadata={
                "loop": {
                    "iterations": loop_output.get("iterations", self._max_iterations),
                    "iterations_executed": loop_output.get("iterations_executed", 0),
                    "terminated_reason": loop_terminated_reason,
                }
            },
        )


__all__ = [
    "ReflexionPattern",
]
