"""Reusable ``plan_execute`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._llm import (
    LLMClient,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents._contracts._tools import ToolRuntime, ToolSpec
from design_research_agents._contracts._workflow import (
    DelegateStep,
    DelegateTarget,
    LogicStep,
    LoopStep,
    ModelStep,
)
from design_research_agents._implementations._agents._multi_step_agent import MultiStepAgent
from design_research_agents._runtime._patterns import (
    MODE_PLAN_EXECUTE,
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_compiled_pattern_execution,
    build_pattern_execution_result,
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
from design_research_agents.workflow import CompiledExecution, Workflow

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


class PlanExecutePattern(Delegate):
    """Planner/executor orchestration pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        planner_delegate: DelegateTarget | None = None,
        executor_delegate: DelegateTarget | None = None,
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
        """Execute one plan-execute orchestration run."""
        return self.compile(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one bound plan-execute orchestration."""
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=self._default_request_id_prefix,
            default_dependencies=self._default_dependencies,
            request_id=request_id,
            dependencies=dependencies,
        )
        resolved_prompt = _extract_prompt(normalize_input_payload(prompt))
        budget_tracker = WorkflowBudgetTracker()
        runtime_state: dict[str, object] = {
            "planner_response": None,
            "parsed_plan": None,
            "plan_steps": [],
            "callbacks": None,
            "fatal_error": None,
            "fatal_stage": None,
            "failure_reason": None,
            "failure_error": None,
        }
        workflow = self._build_workflow(
            resolved_prompt,
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            budget_tracker=budget_tracker,
            runtime_state=runtime_state,
        )
        return build_compiled_pattern_execution(
            workflow=workflow,
            pattern_name="PlanExecutePattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload={"prompt": resolved_prompt, "mode": MODE_PLAN_EXECUTE},
            workflow_request_id=f"{run_context.request_id}:plan_execute_workflow",
            failure_policy="propagate_failed_state",
            finalize=lambda workflow_result: self._build_plan_execute_result(
                workflow_result=workflow_result,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
                budget_tracker=budget_tracker,
                runtime_state=runtime_state,
            ),
        )

    def _build_workflow(
        self,
        prompt: str,
        *,
        request_id: str,
        dependencies: Mapping[str, object],
        budget_tracker: WorkflowBudgetTracker,
        runtime_state: dict[str, object],
    ) -> Workflow:
        """Build the plan/execute workflow for one resolved run context."""
        runtime_tool_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        executor_delegate = self._resolve_executor_delegate()
        planner_step_id = "plan_execute_planner"
        planner_step = self._build_planner_step(
            prompt=prompt,
            planner_step_id=planner_step_id,
        )
        prepare_plan_handler = self._build_plan_prepare_handler(
            prompt=prompt,
            planner_step_id=planner_step_id,
            request_id=request_id,
            dependencies=dependencies,
            budget_tracker=budget_tracker,
            runtime_tool_specs=runtime_tool_specs,
            runtime_state=runtime_state,
        )

        workflow = Workflow(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
            input_schema={"type": "object"},
            base_context={"prompt": prompt},
            steps=[
                planner_step,
                LogicStep(
                    step_id="plan_execute_prepare",
                    handler=prepare_plan_handler,
                    dependencies=(planner_step_id,),
                ),
                LoopStep(
                    step_id="plan_execute_loop",
                    steps=(
                        DelegateStep(
                            step_id="execute_plan_step",
                            delegate=executor_delegate,
                            prompt_builder=self._build_plan_prompt_builder(runtime_state),
                        ),
                    ),
                    dependencies=("plan_execute_prepare",),
                    max_iterations=self._max_iterations,
                    initial_state={"step_results": [], "final_output": {}},
                    continue_predicate=self._build_plan_continue_predicate(runtime_state),
                    state_reducer=self._build_plan_state_reducer(runtime_state),
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                ),
            ],
            default_failure_policy="propagate_failed_state",
        )
        self.workflow = workflow
        return workflow

    def _resolve_executor_delegate(self) -> DelegateTarget:
        """Resolve the executor delegate for plan-step execution."""
        if self._executor_delegate is not None:
            return self._executor_delegate
        return MultiStepAgent(
            mode="code",
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            max_steps=1,
            max_tool_calls_per_step=self._max_tool_calls_per_step,
            tracer=self._tracer,
        )

    def _build_planner_step(
        self,
        *,
        prompt: str,
        planner_step_id: str,
    ) -> ModelStep | DelegateStep:
        """Build the planner step for one compiled execution."""
        if self._planner_delegate is not None:
            planner_prompt = render_prompt_template(
                template_text=self._planner_user_prompt_template,
                variables={"task_prompt": prompt},
                field_name="planner_user_prompt_template",
            )
            return DelegateStep(
                step_id=planner_step_id,
                delegate=self._planner_delegate,
                prompt=planner_prompt,
            )

        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        return ModelStep(
            step_id=planner_step_id,
            llm_client=self._llm_client,
            request_builder=lambda context: self._build_planner_request(
                context=context,
                prompt=prompt,
                resolved_model=resolved_model,
            ),
            response_parser=_parse_planner_model_response,
        )

    def _build_plan_prepare_handler(
        self,
        *,
        prompt: str,
        planner_step_id: str,
        request_id: str,
        dependencies: Mapping[str, object],
        budget_tracker: WorkflowBudgetTracker,
        runtime_tool_specs: Mapping[str, ToolSpec],
        runtime_state: dict[str, object],
    ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
        """Return the planner-output normalization handler for one compiled run."""

        def _prepare_plan_handler(context: Mapping[str, object]) -> Mapping[str, object]:
            self._reset_plan_runtime_state(runtime_state)
            dependency_results = context.get("dependency_results")
            planner_result = (
                dependency_results.get(planner_step_id) if isinstance(dependency_results, Mapping) else None
            )
            if not isinstance(planner_result, Mapping):
                runtime_state["fatal_stage"] = "planner"
                runtime_state["fatal_error"] = "Planner step result is missing."
                return {"plan_valid": False, "plan_steps": []}

            (
                planner_success,
                normalized_metadata,
                planner_response,
                parsed_plan,
            ) = self._extract_planner_response_and_plan(planner_result)

            budget_tracker.add_model_response(planner_response)
            runtime_state["planner_response"] = planner_response

            if not planner_success:
                if self._planner_delegate is None:
                    runtime_state["fatal_stage"] = str(normalized_metadata.get("stage", "planner"))
                    runtime_state["fatal_error"] = str(planner_result.get("error") or "Planner model step failed.")
                    return {"plan_valid": False, "plan_steps": []}
                runtime_state["failure_reason"] = "planner_invalid_json"
                runtime_state["failure_error"] = "Planner did not return valid JSON plan output."
                return {
                    "plan_valid": False,
                    "plan_steps": [],
                    "terminated_reason": "planner_invalid_json",
                }

            if parsed_plan is None:
                runtime_state["failure_reason"] = "planner_invalid_json"
                runtime_state["failure_error"] = "Planner did not return valid JSON plan output."
                return {
                    "plan_valid": False,
                    "plan_steps": [],
                    "terminated_reason": "planner_invalid_json",
                }

            try:
                validate_payload_against_schema(
                    payload=parsed_plan,
                    schema=PLAN_SCHEMA,
                    location="plan_execute.plan",
                )
            except SchemaValidationError as exc:
                runtime_state["parsed_plan"] = dict(parsed_plan)
                runtime_state["failure_reason"] = "planner_invalid_schema"
                runtime_state["failure_error"] = f"Planner output failed schema validation: {exc}"
                return {
                    "plan_valid": False,
                    "plan": dict(parsed_plan),
                    "plan_steps": [],
                    "terminated_reason": "planner_invalid_schema",
                }

            raw_steps = parsed_plan.get("steps")
            plan_steps = (
                [dict(step) for step in raw_steps if isinstance(step, Mapping)] if isinstance(raw_steps, list) else []
            )
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
            runtime_state["parsed_plan"] = dict(parsed_plan)
            runtime_state["plan_steps"] = list(plan_steps)
            runtime_state["callbacks"] = callbacks
            return {
                "plan_valid": True,
                "plan": dict(parsed_plan),
                "plan_steps": list(plan_steps),
            }

        return _prepare_plan_handler

    @staticmethod
    def _reset_plan_runtime_state(runtime_state: dict[str, object]) -> None:
        """Reset mutable plan-execution compile state before one run."""
        runtime_state["planner_response"] = None
        runtime_state["parsed_plan"] = None
        runtime_state["plan_steps"] = []
        runtime_state["callbacks"] = None
        runtime_state["fatal_error"] = None
        runtime_state["fatal_stage"] = None
        runtime_state["failure_reason"] = None
        runtime_state["failure_error"] = None

    def _extract_planner_response_and_plan(
        self,
        planner_result: Mapping[str, object],
    ) -> tuple[bool, dict[str, object], LLMResponse | None, dict[str, object] | None]:
        """Extract planner success metadata, response, and parsed plan payload."""
        planner_output = planner_result.get("output")
        normalized_output = dict(planner_output) if isinstance(planner_output, Mapping) else {}
        planner_success = bool(planner_result.get("success", False))
        planner_metadata = planner_result.get("metadata")
        normalized_metadata = dict(planner_metadata) if isinstance(planner_metadata, Mapping) else {}

        if self._planner_delegate is None:
            planner_response = _extract_model_response_from_model_step_output(normalized_output)
            parsed_plan = self._extract_model_planner_plan(
                normalized_output=normalized_output,
                planner_success=planner_success,
            )
            return planner_success, normalized_metadata, planner_response, parsed_plan

        planner_response = _deserialize_model_response(normalized_output.get("model_response"))
        parsed_plan = self._extract_delegate_planner_plan(
            normalized_output=normalized_output,
            planner_success=planner_success,
        )
        return planner_success, normalized_metadata, planner_response, parsed_plan

    @staticmethod
    def _extract_model_planner_plan(
        *,
        normalized_output: Mapping[str, object],
        planner_success: bool,
    ) -> dict[str, object] | None:
        """Extract a parsed plan from the model-backed planner step output."""
        if not planner_success:
            return None
        parsed_payload = normalized_output.get("parsed")
        if not isinstance(parsed_payload, Mapping):
            return None
        maybe_plan = parsed_payload.get("plan")
        if not isinstance(maybe_plan, Mapping):
            return None
        return dict(maybe_plan)

    @staticmethod
    def _extract_delegate_planner_plan(
        *,
        normalized_output: Mapping[str, object],
        planner_success: bool,
    ) -> dict[str, object] | None:
        """Extract a parsed plan from the delegate-backed planner step output."""
        if not planner_success:
            return None
        nested_output = normalized_output.get("output")
        if not isinstance(nested_output, Mapping):
            return None
        return _extract_planner_payload(nested_output)

    @staticmethod
    def _build_plan_continue_predicate(
        runtime_state: Mapping[str, object],
    ) -> Callable[[int, Mapping[str, object]], bool]:
        """Return the loop continue predicate for one compiled plan-execute run."""

        def _continue_predicate(iteration: int, state: Mapping[str, object]) -> bool:
            del state
            plan_steps = runtime_state.get("plan_steps")
            return iteration <= len(plan_steps) if isinstance(plan_steps, list) else False

        return _continue_predicate

    @staticmethod
    def _build_plan_prompt_builder(
        runtime_state: Mapping[str, object],
    ) -> Callable[[Mapping[str, object]], str]:
        """Return the loop prompt builder for one compiled plan-execute run."""

        def _executor_prompt_builder(step_context: Mapping[str, object]) -> str:
            callbacks = runtime_state.get("callbacks")
            if not isinstance(callbacks, PlanExecuteLoopCallbacks):
                raise RuntimeError("Plan execute callbacks are unavailable for loop execution.")
            return callbacks.executor_prompt_builder(step_context)

        return _executor_prompt_builder

    @staticmethod
    def _build_plan_state_reducer(
        runtime_state: Mapping[str, object],
    ) -> Callable[[Mapping[str, object], ExecutionResult, int], Mapping[str, object]]:
        """Return the loop state reducer for one compiled plan-execute run."""

        def _state_reducer(
            state: Mapping[str, object],
            iteration_result: ExecutionResult,
            iteration: int,
        ) -> Mapping[str, object]:
            callbacks = runtime_state.get("callbacks")
            if not isinstance(callbacks, PlanExecuteLoopCallbacks):
                return dict(state)
            return callbacks.state_reducer(state, iteration_result, iteration)

        return _state_reducer

    def _run_plan_execute(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Backwards-compatible internal wrapper over the compile-first path."""
        return self.compile(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def _build_plan_execute_result(
        self,
        *,
        workflow_result: ExecutionResult,
        request_id: str,
        dependencies: Mapping[str, object],
        budget_tracker: WorkflowBudgetTracker,
        runtime_state: Mapping[str, object],
    ) -> ExecutionResult:
        """Build the finalized plan-execute result from compiled workflow state."""
        fatal_error = runtime_state.get("fatal_error")
        if isinstance(fatal_error, str) and fatal_error:
            fatal_stage = str(runtime_state.get("fatal_stage", "planner"))
            if fatal_stage == "input_build":
                raise ValueError(fatal_error)
            raise RuntimeError(fatal_error)

        planner_response = runtime_state.get("planner_response")
        model_response = planner_response if isinstance(planner_response, LLMResponse) else None
        parsed_plan_value = runtime_state.get("parsed_plan")
        parsed_plan = dict(parsed_plan_value) if isinstance(parsed_plan_value, Mapping) else None
        failure_reason = runtime_state.get("failure_reason")
        failure_error = runtime_state.get("failure_error")

        if isinstance(failure_reason, str) and failure_reason:
            failure = build_pattern_execution_result(
                success=False,
                final_output={},
                terminated_reason=failure_reason,
                details={
                    "plan": parsed_plan,
                    "plan_schema_version": PLAN_SCHEMA_VERSION,
                    "steps_executed": 0,
                    "step_results": [],
                },
                workflow_payload=workflow_result.to_dict(),
                artifacts=workflow_result.output.get("artifacts", []),
                request_id=request_id,
                dependencies=dependencies,
                mode=MODE_PLAN_EXECUTE,
                metadata={"stage": "planner"},
                tool_results=[],
                model_response=model_response,
                error=(
                    str(failure_error)
                    if isinstance(failure_error, str) and failure_error
                    else "Planner did not return a valid execution plan."
                ),
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode=MODE_PLAN_EXECUTE,
                resolved_mode=MODE_PLAN_EXECUTE,
                budget_metadata=budget_tracker.as_metadata(),
                extra_metadata=None,
            )

        callbacks = runtime_state.get("callbacks")
        if not isinstance(callbacks, PlanExecuteLoopCallbacks) or parsed_plan is None:
            raise RuntimeError("Plan execute runtime state is unavailable before workflow execution.")

        plan_steps_value = runtime_state.get("plan_steps")
        plan_steps = (
            [dict(step) for step in plan_steps_value if isinstance(step, Mapping)]
            if isinstance(plan_steps_value, list)
            else []
        )
        loop_step_result = workflow_result.step_results.get("plan_execute_loop")
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
            workflow_payload=workflow_result.to_dict(),
            artifacts=workflow_result.output.get("artifacts", []),
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


def _deserialize_model_response(raw_model_response: object) -> LLMResponse | None:
    """Deserialize one serialized model-response mapping."""
    if not isinstance(raw_model_response, Mapping):
        return None
    try:
        return LLMResponse(**dict(raw_model_response))
    except TypeError:
        return None


__all__ = [
    "PlanExecutePattern",
]
