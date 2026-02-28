"""Reusable ``plan_execute`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents._contracts._agent import Agent, ExecutionResult
from design_research_agents._contracts._llm import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents._contracts._tools import ToolRuntime
from design_research_agents._contracts._workflow import (
    AgentStep,
    LogicStep,
    LoopStep,
    ModelStep,
    WorkflowDelegate,
)
from design_research_agents._implementations._agents._multi_step_agent import MultiStepAgent
from design_research_agents._runtime._common._delegate_invocation import invoke_delegate
from design_research_agents._runtime._patterns import (
    MODE_PLAN_EXECUTE,
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_pattern_execution_result,
    execute_pattern_with_trace,
    normalize_mapping,
    normalize_mapping_records,
    normalize_request_id_prefix,
    render_prompt_template,
    resolve_pattern_run_context,
    resolve_prompt_override,
)
from design_research_agents._schemas import (
    SchemaValidationError,
    validate_payload_against_schema,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow.workflow import Workflow

from .._shared._agent_internal._input_parsing import (
    extract_prompt as _extract_prompt,
)
from .._shared._agent_internal._input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from .._shared._agent_internal._model_resolution import resolve_agent_model
from .._shared._agent_internal._run_options import normalize_input_payload
from .._shared._workflow_internal._plan_execute_helpers import (
    DEFAULT_EXECUTOR_STEP_PROMPT_TEMPLATE,
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    DEFAULT_PLANNER_USER_PROMPT_TEMPLATE,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    PlanExecuteLoopCallbacks,
)


class PlanExecutePattern(Agent):
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
        planner_system_prompt: str | None = None,
        planner_user_prompt_template: str | None = None,
        executor_step_prompt_template: str | None = None,
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
            planner_system_prompt: Optional override for planner system prompt.
            planner_user_prompt_template: Optional override for planner user prompt.
            executor_step_prompt_template: Optional override for executor step prompt.
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
        self._plan_execute_runtime: dict[str, object] | None = None
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._planner_system_prompt = resolve_prompt_override(
            override=planner_system_prompt,
            default_value=DEFAULT_PLANNER_SYSTEM_PROMPT,
            field_name="planner_system_prompt",
        )
        self._planner_user_prompt_template = resolve_prompt_override(
            override=planner_user_prompt_template,
            default_value=DEFAULT_PLANNER_USER_PROMPT_TEMPLATE,
            field_name="planner_user_prompt_template",
        )
        self._executor_step_prompt_template = resolve_prompt_override(
            override=executor_step_prompt_template,
            default_value=DEFAULT_EXECUTOR_STEP_PROMPT_TEMPLATE,
            field_name="executor_step_prompt_template",
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
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=self._default_request_id_prefix,
            default_dependencies=self._default_dependencies,
            request_id=request_id,
            dependencies=dependencies,
        )
        resolved_prompt = _extract_prompt(normalize_input_payload(prompt))
        return execute_pattern_with_trace(
            agent_name="PlanExecutePattern",
            request_id=run_context.request_id,
            input_payload={"prompt": resolved_prompt, "mode": MODE_PLAN_EXECUTE},
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            runner=lambda: self._run_plan_execute(
                prompt=resolved_prompt,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def build_workflow(
        self,
        prompt: str,
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> Workflow:
        """Build the plan/execute workflow for one resolved run context."""
        budget_tracker = WorkflowBudgetTracker()
        runtime_tool_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        planner_response: LLMResponse | None = None
        parsed_plan: dict[str, object] | None = None

        if self._planner_delegate is None:
            parsed_plan, planner_response = self._run_planner_model_step(
                prompt=prompt,
                request_id=request_id,
                dependencies=dependencies,
            )
            budget_tracker.add_model_response(planner_response)
        else:
            planner_prompt = render_prompt_template(
                template_text=self._planner_user_prompt_template,
                variables={"task_prompt": prompt},
                field_name="planner_user_prompt_template",
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

        failure_result: ExecutionResult | None = None
        if parsed_plan is None:
            failure = build_pattern_execution_result(
                success=False,
                final_output={},
                terminated_reason="planner_invalid_json",
                details={
                    "plan": None,
                    "plan_schema_version": PLAN_SCHEMA_VERSION,
                    "steps_executed": 0,
                    "step_results": [],
                },
                workflow_payload={},
                artifacts=[],
                request_id=request_id,
                dependencies=dependencies,
                mode=MODE_PLAN_EXECUTE,
                metadata={"stage": "planner"},
                tool_results=[],
                model_response=planner_response,
                error="Planner did not return valid JSON plan output.",
            )
            failure_result = attach_runtime_metadata(
                agent_result=failure,
                requested_mode=MODE_PLAN_EXECUTE,
                resolved_mode=MODE_PLAN_EXECUTE,
                budget_metadata=budget_tracker.as_metadata(),
                extra_metadata=None,
            )
        else:
            try:
                validate_payload_against_schema(
                    payload=parsed_plan,
                    schema=PLAN_SCHEMA,
                    location="plan_execute.plan",
                )
            except SchemaValidationError as exc:
                failure = build_pattern_execution_result(
                    success=False,
                    final_output={},
                    terminated_reason="planner_invalid_schema",
                    details={
                        "plan": parsed_plan,
                        "plan_schema_version": PLAN_SCHEMA_VERSION,
                        "steps_executed": 0,
                        "step_results": [],
                    },
                    workflow_payload={},
                    artifacts=[],
                    request_id=request_id,
                    dependencies=dependencies,
                    mode=MODE_PLAN_EXECUTE,
                    metadata={"stage": "planner"},
                    tool_results=[],
                    model_response=planner_response,
                    error=f"Planner output failed schema validation: {exc}",
                )
                failure_result = attach_runtime_metadata(
                    agent_result=failure,
                    requested_mode=MODE_PLAN_EXECUTE,
                    resolved_mode=MODE_PLAN_EXECUTE,
                    budget_metadata=budget_tracker.as_metadata(),
                    extra_metadata=None,
                )

        if failure_result is not None:
            workflow = Workflow(
                tool_runtime=None,
                tracer=self._tracer,
                input_schema={"type": "object"},
                steps=[
                    LogicStep(
                        step_id="plan_execute_noop",
                        handler=lambda context: {},
                    )
                ],
            )
            self.workflow = workflow
            self._plan_execute_runtime = {"failure": failure_result}
            return workflow

        assert parsed_plan is not None
        raw_steps = parsed_plan.get("steps")
        plan_steps = (
            [dict(step) for step in raw_steps if isinstance(step, Mapping)] if isinstance(raw_steps, list) else []
        )

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
            plan_steps=plan_steps,
            executor_step_prompt_template=self._executor_step_prompt_template,
            request_id=request_id,
            dependencies=dependencies,
            budget_tracker=budget_tracker,
            runtime_tool_specs=runtime_tool_specs,
            initial_model_response=planner_response,
        )

        workflow = Workflow(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
            input_schema={"type": "object"},
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
        self.workflow = workflow
        self._plan_execute_runtime = {
            "failure": None,
            "budget_tracker": budget_tracker,
            "callbacks": callbacks,
            "parsed_plan": dict(parsed_plan),
            "plan_steps": list(plan_steps),
        }
        return workflow

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
        workflow = self.build_workflow(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        runtime = self._plan_execute_runtime or {}
        failure = runtime.get("failure")
        if isinstance(failure, ExecutionResult):
            return failure
        budget_tracker = runtime.get("budget_tracker")
        callbacks = runtime.get("callbacks")
        parsed_plan_value = runtime.get("parsed_plan")
        plan_steps_value = runtime.get("plan_steps")
        if (
            not isinstance(budget_tracker, WorkflowBudgetTracker)
            or not isinstance(callbacks, PlanExecuteLoopCallbacks)
            or not isinstance(parsed_plan_value, Mapping)
            or not isinstance(plan_steps_value, list)
        ):
            raise RuntimeError("Plan execute runtime state is unavailable before workflow execution.")
        parsed_plan = dict(parsed_plan_value)
        plan_steps = [dict(step) for step in plan_steps_value if isinstance(step, Mapping)]

        loop_workflow_result = workflow.run(
            input={},
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
            terminated_reason = "truncated_max_iterations"
        else:
            terminated_reason = "completed"

        plan_execute_result = build_pattern_execution_result(
            success=terminated_reason in {"completed", "truncated_max_iterations"} and bool(step_results),
            final_output=final_output,
            terminated_reason=terminated_reason,
            details={
                "plan": parsed_plan,
                "plan_schema_version": PLAN_SCHEMA_VERSION,
                "steps_executed": len(step_results),
                "step_results": step_results,
            },
            workflow_payload=loop_workflow_result.to_dict(),
            artifacts=loop_workflow_result.output.get("artifacts", []),
            request_id=request_id,
            dependencies=dependencies,
            mode=MODE_PLAN_EXECUTE,
            metadata={"stage": "execution"},
            tool_results=callbacks.all_tool_results,
            model_response=callbacks.last_model_response,
        )
        return attach_runtime_metadata(
            agent_result=plan_execute_result,
            requested_mode=MODE_PLAN_EXECUTE,
            resolved_mode=MODE_PLAN_EXECUTE,
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

    def _run_planner_model_step(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> tuple[dict[str, object] | None, LLMResponse | None]:
        """Run planner model call through ``ModelStep`` and extract parsed plan output."""
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        planner_workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            base_context={"prompt": prompt},
            steps=[
                ModelStep(
                    step_id="plan_execute_planner_model",
                    llm_client=self._llm_client,
                    request_builder=lambda context: self._build_planner_request(
                        context=context,
                        prompt=prompt,
                        resolved_model=resolved_model,
                    ),
                    response_parser=_parse_planner_model_response,
                )
            ],
        )
        planner_result = planner_workflow.run(
            input={},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:plan_execute_planner_model",
            dependencies=dependencies,
        )
        planner_step = planner_result.step_results.get("plan_execute_planner_model")
        if planner_step is None:
            raise RuntimeError("Planner model step result is missing.")
        if not planner_step.success:
            error_text = planner_step.error or "Planner model step failed."
            stage = str(planner_step.metadata.get("stage", ""))
            if stage == "input_build":
                raise ValueError(error_text)
            raise RuntimeError(error_text)

        planner_response = _extract_model_response_from_model_step_output(planner_step.output)
        parsed_payload = planner_step.output.get("parsed")
        if not isinstance(parsed_payload, Mapping):
            return None, planner_response
        plan_payload = parsed_payload.get("plan")
        if not isinstance(plan_payload, Mapping):
            return None, planner_response
        return dict(plan_payload), planner_response

    def _build_planner_request(
        self,
        *,
        context: Mapping[str, object],
        prompt: str,
        resolved_model: str,
    ) -> LLMRequest:
        """Build one planner ``LLMRequest`` payload for ``ModelStep`` execution."""
        del context
        planner_prompt = render_prompt_template(
            template_text=self._planner_user_prompt_template,
            variables={"task_prompt": prompt},
            field_name="planner_user_prompt_template",
        )
        planner_metadata: dict[str, object] = {
            "agent": "PlanExecutePattern",
            "mode": MODE_PLAN_EXECUTE,
            "phase": "planner",
        }
        planner_messages = [
            LLMMessage(role="system", content=self._planner_system_prompt),
            LLMMessage(role="user", content=planner_prompt),
        ]
        return LLMRequest(
            messages=planner_messages,
            model=resolved_model,
            response_schema=dict(PLAN_SCHEMA),
            metadata=dict(planner_metadata),
            provider_options=dict(planner_metadata),
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


def _parse_planner_model_response(
    response: LLMResponse,
    context: Mapping[str, object],
) -> Mapping[str, object]:
    """Parse planner model response into ``{"plan": ...}`` payload."""
    del context
    return {"plan": _parse_json_mapping(response.text)}


def _extract_model_response_from_model_step_output(
    output: Mapping[str, object],
) -> LLMResponse | None:
    """Extract ``LLMResponse`` from serialized ``ModelStep`` output payload."""
    raw_model_response = output.get("model_response")
    if not isinstance(raw_model_response, Mapping):
        return None
    try:
        return LLMResponse(**dict(raw_model_response))
    except TypeError:
        return None


__all__ = [
    "PlanExecutePattern",
]
