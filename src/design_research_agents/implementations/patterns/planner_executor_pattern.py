"""Reusable ``plan_execute`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import AgentStep, LoopStep, WorkflowDelegate
from design_research_agents.implementations.agents.multi_step_agent import MultiStepAgent
from design_research_agents.schemas import (
    SchemaValidationError,
    validate_payload_against_schema,
)
from design_research_agents.tracing import (
    Tracer,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)
from design_research_agents.workflow.workflow import Workflow

from ..shared.agent_internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from ..shared.agent_internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from ..shared.agent_internal.model_resolution import resolve_agent_model
from ..shared.agent_internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from ..shared.workflow_internal import (
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_pattern_failure_result,
    merge_dependencies,
    normalize_mapping,
    normalize_mapping_records,
    normalize_request_id_prefix,
    render_prompt_template,
    resolve_prompt_override,
    resolve_request_id_with_prefix,
)
from ..shared.workflow_internal.delegate_invocation import invoke_delegate
from ..shared.workflow_internal.planner_executor_helpers import (
    DEFAULT_EXECUTOR_STEP_PROMPT_TEMPLATE,
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    DEFAULT_PLANNER_USER_PROMPT_TEMPLATE,
    PLAN_SCHEMA,
    PlanExecuteLoopCallbacks,
)


class PlannerExecutorPattern(Agent):
    """Planner/executor orchestration pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        planner_delegate: WorkflowDelegate | None = None,
        executor_delegate: WorkflowDelegate | None = None,
        max_iterations: int = 3,
        max_tool_calls_per_step: int = 5,
        plan_execute_planner_system_prompt: str | None = None,
        plan_execute_planner_user_prompt_template: str | None = None,
        plan_execute_executor_step_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize workflow-native orchestration settings.

        Args:
            llm_client: LLM client used for planner and executor model calls.
            tool_runtime: Tool runtime used by executor agent steps.
            planner_delegate: Optional planner delegate override.
            executor_delegate: Optional executor delegate override.
            max_iterations: Maximum number of plan steps executed in one run.
            max_tool_calls_per_step: Maximum tool calls allowed per executor step.
            plan_execute_planner_system_prompt: Optional override for planner system prompt.
            plan_execute_planner_user_prompt_template: Optional override for planner user prompt.
            plan_execute_executor_step_prompt_template: Optional override for executor step prompt.
            default_request_id_prefix: Optional prefix used to derive request ids.
            default_dependencies: Dependency defaults merged into each run.
            tracer: Optional tracer used for run-level instrumentation.

        Raises:
            ValueError: If ``max_iterations`` or ``max_tool_calls_per_step`` is invalid.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        if max_tool_calls_per_step < 1:
            raise ValueError("max_tool_calls_per_step must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._planner_delegate = planner_delegate
        self._executor_delegate = executor_delegate
        self._max_iterations = max_iterations
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._tracer = tracer
        self.workflow: Workflow | None = None
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._planner_system_prompt = resolve_prompt_override(
            override=plan_execute_planner_system_prompt,
            default_value=DEFAULT_PLANNER_SYSTEM_PROMPT,
            field_name="plan_execute_planner_system_prompt",
        )
        self._planner_user_prompt_template = resolve_prompt_override(
            override=plan_execute_planner_user_prompt_template,
            default_value=DEFAULT_PLANNER_USER_PROMPT_TEMPLATE,
            field_name="plan_execute_planner_user_prompt_template",
        )
        self._executor_step_prompt_template = resolve_prompt_override(
            override=plan_execute_executor_step_prompt_template,
            default_value=DEFAULT_EXECUTOR_STEP_PROMPT_TEMPLATE,
            field_name="plan_execute_executor_step_prompt_template",
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one plan-execute orchestration run.

        Args:
            prompt: Task prompt to plan and execute.
            request_id: Optional request id for tracing and correlation.
            dependencies: Optional dependency overrides for this run.

        Returns:
            Pattern result containing plan, step results, and final output.

        Raises:
            Exception: Propagates runtime failures from planner or executor phases.
        """
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
        resolved_prompt = _extract_prompt(normalize_input_payload(prompt))
        trace_scope = start_trace_run(
            agent_name="PlannerExecutorPattern",
            request_id=resolved_request_id,
            input_payload={"prompt": resolved_prompt, "mode": "plan_execute"},
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )

        try:
            result = self._run_plan_execute(
                prompt=resolved_prompt,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        finish_trace_run(trace_scope, result=result)
        return result

    def _run_plan_execute(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Run planner phase then execute planned steps through a loop step.

        Args:
            prompt: Task prompt to plan and execute.
            request_id: Resolved request id for this orchestration run.
            dependencies: Normalized dependency mapping for this run.

        Returns:
            Final plan-execute pattern result.

        Raises:
            RuntimeError: If required loop outputs are missing.
        """
        budget_tracker = WorkflowBudgetTracker()
        runtime_tool_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        planner_response: LLMResponse | None = None
        parsed_plan: dict[str, object] | None = None

        if self._planner_delegate is None:
            resolved_model = resolve_agent_model(llm_client=self._llm_client)
            planner_messages = [
                LLMMessage(role="system", content=self._planner_system_prompt),
                LLMMessage(
                    role="user",
                    content=render_prompt_template(
                        template_text=self._planner_user_prompt_template,
                        variables={"task_prompt": prompt},
                        field_name="plan_execute_planner_user_prompt_template",
                    ),
                ),
            ]
            planner_call_metadata: dict[str, object] = {
                "agent": "PlannerExecutorPattern",
                "mode": "plan_execute",
                "phase": "planner",
            }
            planner_params = LLMChatParams(
                response_schema=dict(PLAN_SCHEMA),
                provider_options=planner_call_metadata,
            )
            planner_span_id = start_model_call(
                model=resolved_model,
                messages=planner_messages,
                params=planner_params,
                metadata=planner_call_metadata,
            )
            try:
                planner_response = self._llm_client.chat(
                    planner_messages,
                    model=resolved_model,
                    params=planner_params,
                )
            except Exception as exc:
                finish_model_call(planner_span_id, error=str(exc), model=resolved_model)
                raise
            finish_model_call(planner_span_id, response=planner_response)
            budget_tracker.add_model_response(planner_response)
            parsed_plan = _parse_json_mapping(planner_response.text)
        else:
            planner_prompt = render_prompt_template(
                template_text=self._planner_user_prompt_template,
                variables={"task_prompt": prompt},
                field_name="plan_execute_planner_user_prompt_template",
            )
            planner_invocation = invoke_delegate(
                delegate=self._planner_delegate,
                prompt=planner_prompt,
                step_context=None,
                request_id=f"{request_id}:plan_execute:planner_delegate",
                execution_mode="sequential",
                failure_policy="skip_dependents",
                dependencies=dependencies,
            )
            planner_result = planner_invocation.result
            planner_response = planner_result.model_response
            budget_tracker.add_model_response(planner_response)
            if planner_result.success:
                parsed_plan = _extract_planner_payload(planner_result.output)

        if parsed_plan is None:
            failure = build_pattern_failure_result(
                error="Planner did not return valid JSON plan output.",
                model_response=planner_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={"stage": "planner", "mode": "plan_execute"},
                output={
                    "terminated_reason": "planner_invalid_json",
                    "plan": None,
                    "steps_executed": 0,
                    "step_results": [],
                    "final_output": {},
                },
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode="plan_execute",
                resolved_mode="plan_execute",
                budget_metadata=budget_tracker.as_metadata(),
                extra_metadata=None,
            )

        try:
            validate_payload_against_schema(
                payload=parsed_plan,
                schema=PLAN_SCHEMA,
                location="plan_execute.plan",
            )
        except SchemaValidationError as exc:
            failure = build_pattern_failure_result(
                error=f"Planner output failed schema validation: {exc}",
                model_response=planner_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={"stage": "planner", "mode": "plan_execute"},
                output={
                    "terminated_reason": "planner_invalid_schema",
                    "plan": parsed_plan,
                    "steps_executed": 0,
                    "step_results": [],
                    "final_output": {},
                },
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode="plan_execute",
                resolved_mode="plan_execute",
                budget_metadata=budget_tracker.as_metadata(),
                extra_metadata=None,
            )

        raw_steps = parsed_plan.get("steps")
        plan_steps = raw_steps if isinstance(raw_steps, list) else []

        if self._executor_delegate is None:
            executor_delegate: WorkflowDelegate = MultiStepAgent(
                mode="code",
                llm_client=self._llm_client,
                tool_runtime=self._tool_runtime,
                max_steps=1,
                max_tool_calls_per_step=self._max_tool_calls_per_step,
                tracer=self._tracer,
            )
        else:
            executor_delegate = self._executor_delegate
        callbacks = PlanExecuteLoopCallbacks(
            prompt=prompt,
            plan_steps=[dict(step) for step in plan_steps if isinstance(step, Mapping)],
            executor_step_prompt_template=self._executor_step_prompt_template,
            request_id=request_id,
            dependencies=dependencies,
            budget_tracker=budget_tracker,
            runtime_tool_specs=runtime_tool_specs,
            initial_model_response=planner_response,
        )

        self.workflow = Workflow(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
            input_mode="schema",
            base_context={"prompt": prompt},
            steps=[
                LoopStep(
                    step_id="plan_execute_loop",
                    steps=(
                        AgentStep(
                            step_id="execute_plan_step",
                            delegate=executor_delegate,
                            prompt_builder=callbacks.executor_prompt_builder,
                        ),
                    ),
                    max_iterations=self._max_iterations,
                    initial_state={"step_results": [], "final_output": {}},
                    continue_predicate=callbacks.continue_predicate,
                    state_reducer=callbacks.state_reducer,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                )
            ],
        )

        loop_workflow_result = self.workflow.run(
            {},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:plan_execute_loop",
            dependencies=dependencies,
        )
        loop_step_result = loop_workflow_result.step_results.get("plan_execute_loop")
        if loop_step_result is None:
            raise RuntimeError("Plan execute loop step result is missing.")

        loop_output = loop_step_result.output
        final_state = normalize_mapping(loop_output.get("final_state"))
        step_results = normalize_mapping_records(final_state.get("step_results"))
        final_output: dict[str, object] = {}
        maybe_final_output = final_state.get("final_output")
        if isinstance(maybe_final_output, Mapping):
            final_output = dict(maybe_final_output)

        loop_terminated_reason = str(loop_output.get("terminated_reason", "max_iterations_reached"))

        if loop_terminated_reason == "iteration_failed":
            terminated_reason = "step_failure"
        elif len(plan_steps) > self._max_iterations:
            terminated_reason = "max_iterations_reached"
        else:
            terminated_reason = "completed"

        plan_execute_result = ExecutionResult(
            output={
                "plan": parsed_plan,
                "steps_executed": len(step_results),
                "step_results": step_results,
                "final_output": final_output,
                "terminated_reason": terminated_reason,
                "workflow": loop_workflow_result.asdict(),
                "artifacts": loop_workflow_result.output.get("artifacts", []),
            },
            success=terminated_reason in {"completed", "max_iterations_reached"}
            and bool(step_results),
            tool_results=callbacks.all_tool_results,
            model_response=callbacks.last_model_response,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "stage": "execution",
                "mode": "plan_execute",
            },
        )
        return attach_runtime_metadata(
            agent_result=plan_execute_result,
            requested_mode="plan_execute",
            resolved_mode="plan_execute",
            budget_metadata=budget_tracker.as_metadata(),
            extra_metadata={
                "plan": {
                    "step_count": len(plan_steps),
                    "executed_step_count": len(step_results),
                },
                "loop": {
                    "iterations": loop_output.get("iterations", self._max_iterations),
                    "iterations_executed": loop_output.get("iterations_executed", 0),
                    "terminated_reason": loop_terminated_reason,
                },
            },
        )


def _extract_planner_payload(output: Mapping[str, object]) -> dict[str, object] | None:
    """Extract planner payload mapping from delegate output.

    Args:
        output: Delegate output payload.

    Returns:
        Planner payload mapping when present, otherwise ``None``.
    """
    steps = output.get("steps")
    if isinstance(steps, list):
        return dict(output)

    final_output = output.get("final_output")
    if isinstance(final_output, Mapping):
        final_steps = final_output.get("steps")
        if isinstance(final_steps, list):
            return dict(final_output)
    if isinstance(final_output, str):
        parsed_final = _parse_json_mapping(final_output)
        if parsed_final is not None:
            return parsed_final

    model_text = output.get("model_text")
    if isinstance(model_text, str):
        return _parse_json_mapping(model_text)
    return None


__all__ = [
    "PlannerExecutorPattern",
]
