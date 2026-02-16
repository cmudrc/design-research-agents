"""Reusable ``plan_execute`` orchestration chunk."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

from design_research_agents.agent.implementations.single_step_code_tool_calling_agent import (
    SingleStepCodeToolCallingAgent,
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
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime
from design_research_agents.contracts.workflow import AgentStep, LoopStep, WorkflowResult
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

_PLAN_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_id", "instruction", "success_criteria"],
                "properties": {
                    "step_id": {"type": "string"},
                    "instruction": {"type": "string"},
                    "success_criteria": {"type": "string"},
                },
            },
        }
    },
}

_DEFAULT_PLAN_EXECUTE_PLANNER_SYSTEM_PROMPT = (
    "You are a planner for a plan-execute runtime. Return strict JSON only with steps[]."
)
_DEFAULT_PLAN_EXECUTE_PLANNER_USER_PROMPT_TEMPLATE = (
    "Create an execution plan for this task. "
    "Each step must have step_id, instruction, and success_criteria.\n\n"
    "Task:\n$task_prompt"
)
_DEFAULT_PLAN_EXECUTE_EXECUTOR_STEP_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Plan step id: $step_id",
        "Instruction: $instruction",
        "Success criteria: $success_criteria",
        "Prior step outputs:",
        "$prior_step_outputs_json",
    ]
)


class PlannerExecutorPattern(Agent):
    """Planner/executor orchestration pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        controls: RuntimeControls | None = None,
        plan_execute_planner_system_prompt: str | None = None,
        plan_execute_planner_user_prompt_template: str | None = None,
        plan_execute_executor_step_prompt_template: str | None = None,
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
        self._planner_system_prompt = resolve_prompt_override(
            override=plan_execute_planner_system_prompt,
            default_value=_DEFAULT_PLAN_EXECUTE_PLANNER_SYSTEM_PROMPT,
            field_name="plan_execute_planner_system_prompt",
        )
        self._planner_user_prompt_template = resolve_prompt_override(
            override=plan_execute_planner_user_prompt_template,
            default_value=_DEFAULT_PLAN_EXECUTE_PLANNER_USER_PROMPT_TEMPLATE,
            field_name="plan_execute_planner_user_prompt_template",
        )
        self._executor_step_prompt_template = resolve_prompt_override(
            override=plan_execute_executor_step_prompt_template,
            default_value=_DEFAULT_PLAN_EXECUTE_EXECUTOR_STEP_PROMPT_TEMPLATE,
            field_name="plan_execute_executor_step_prompt_template",
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one plan-execute orchestration run."""
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

    def _run_plan_execute(  # noqa: C901
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> AgentResult:
        budget_tracker = WorkflowBudgetTracker()
        runtime_tool_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
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
        planner_params = LLMChatParams(
            response_schema=dict(_PLAN_SCHEMA),
            provider_options={
                "agent": "PlannerExecutorPattern",
                "mode": "plan_execute",
                "phase": "planner",
            },
        )
        planner_span_id = start_model_call(
            model=resolved_model,
            messages=planner_messages,
            params=planner_params,
            metadata={
                "agent": "PlannerExecutorPattern",
                "mode": "plan_execute",
                "phase": "planner",
            },
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
                controls=self._controls,
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        try:
            validate_payload_against_schema(
                payload=parsed_plan,
                schema=_PLAN_SCHEMA,
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
                controls=self._controls,
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        raw_steps = parsed_plan.get("steps")
        plan_steps = raw_steps if isinstance(raw_steps, list) else []

        executor_agent = SingleStepCodeToolCallingAgent(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            max_tool_calls=self._controls.max_tool_calls_per_step,
            execution_timeout_seconds=self._controls.execution_timeout_seconds_per_step,
            tracer=self._tracer,
        )
        workflow_runtime = WorkflowRuntime(
            tool_runtime=self._tool_runtime,
            agents={"plan_executor": executor_agent},
            tracer=self._tracer,
        )

        all_tool_results: list[ToolResult] = []
        step_results: list[dict[str, object]] = []
        final_output: dict[str, object] = {}
        last_model_response: LLMResponse | None = planner_response

        def _continue_predicate(iteration: int, state: Mapping[str, object]) -> bool:
            del state
            return iteration <= len(plan_steps)

        def _executor_prompt_builder(step_context: Mapping[str, object]) -> str:
            loop_metadata = step_context.get("_loop")
            if not isinstance(loop_metadata, Mapping):
                raise ValueError("Loop metadata is required for plan execution prompt building.")

            raw_iteration = loop_metadata.get("iteration")
            iteration = parse_loop_iteration(
                raw_iteration,
                error_prefix="Plan execute loop iteration",
            )

            if iteration < 1 or iteration > len(plan_steps):
                raise ValueError(f"Loop iteration {iteration} is out of bounds for planned steps.")

            raw_step = plan_steps[iteration - 1]
            if not isinstance(raw_step, Mapping):
                raise ValueError(f"Plan step at iteration {iteration} is not a mapping.")

            loop_state = step_context.get("loop_state")
            loop_step_records = (
                normalize_mapping_records(loop_state.get("step_results"))
                if isinstance(loop_state, Mapping)
                else []
            )

            step_id = str(raw_step.get("step_id", f"step_{iteration}"))
            step_instruction = str(raw_step.get("instruction", ""))
            success_criteria = str(raw_step.get("success_criteria", ""))
            step_prompt = render_prompt_template(
                template_text=self._executor_step_prompt_template,
                variables={
                    "task_prompt": prompt,
                    "step_id": step_id,
                    "instruction": step_instruction,
                    "success_criteria": success_criteria,
                    "prior_step_outputs_json": json.dumps(
                        loop_step_records[-3:],
                        sort_keys=True,
                    ),
                },
                field_name="plan_execute_executor_step_prompt_template",
            )
            return step_prompt

        def _state_reducer(
            state: Mapping[str, object],
            iteration_result: WorkflowResult,
            iteration: int,
        ) -> Mapping[str, object]:
            nonlocal final_output
            nonlocal last_model_response

            raw_step = plan_steps[iteration - 1]
            if not isinstance(raw_step, Mapping):
                return dict(state)

            state_step_results = normalize_mapping_records(state.get("step_results"))
            state_final_output = normalize_mapping(state.get("final_output"))

            step_id = str(raw_step.get("step_id", f"step_{iteration}"))
            step_instruction = str(raw_step.get("instruction", ""))
            success_criteria = str(raw_step.get("success_criteria", ""))
            execution_step = iteration_result.step_results.get("execute_plan_step")

            if execution_step is None:
                state_step_results.append(
                    {
                        "step_id": step_id,
                        "instruction": step_instruction,
                        "success_criteria": success_criteria,
                        "success": False,
                        "final_output": {},
                        "error": "Workflow iteration did not include execute_plan_step result.",
                    }
                )
                return {
                    "step_results": state_step_results,
                    "final_output": state_final_output,
                }

            serialized_agent_result = (
                execution_step.output if isinstance(execution_step.output, Mapping) else {}
            )
            serialized_output = serialized_agent_result.get("output")
            agent_output = dict(serialized_output) if isinstance(serialized_output, Mapping) else {}

            step_tool_results = _deserialize_tool_results(
                serialized_agent_result.get("tool_results")
            )
            step_model_response = _deserialize_model_response(
                serialized_agent_result.get("model_response")
            )
            budget_tracker.add_model_response(step_model_response)
            budget_tracker.add_tool_results(
                tool_results=step_tool_results,
                tool_specs=runtime_tool_specs,
            )
            if step_model_response is not None:
                last_model_response = step_model_response
            all_tool_results.extend(step_tool_results)

            maybe_step_success = serialized_agent_result.get("success")
            if isinstance(maybe_step_success, bool):
                step_success = maybe_step_success
            else:
                step_success = execution_step.success

            maybe_final_output = agent_output.get("final_output")
            normalized_final_output = (
                dict(maybe_final_output) if isinstance(maybe_final_output, Mapping) else {}
            )
            maybe_error = agent_output.get("error")
            normalized_error = str(maybe_error) if maybe_error is not None else execution_step.error

            state_step_results.append(
                {
                    "step_id": step_id,
                    "instruction": step_instruction,
                    "success_criteria": success_criteria,
                    "success": step_success,
                    "final_output": normalized_final_output,
                    "error": normalized_error,
                }
            )
            if step_success and isinstance(maybe_final_output, Mapping):
                state_final_output = dict(maybe_final_output)

            return {
                "step_results": state_step_results,
                "final_output": state_final_output,
            }

        loop_workflow_result = workflow_runtime.run(
            steps=[
                LoopStep(
                    step_id="plan_execute_loop",
                    steps=(
                        AgentStep(
                            step_id="execute_plan_step",
                            agent_name="plan_executor",
                            prompt_builder=_executor_prompt_builder,
                        ),
                    ),
                    max_iterations=self._controls.max_iterations,
                    initial_state={"step_results": [], "final_output": {}},
                    continue_predicate=_continue_predicate,
                    state_reducer=_state_reducer,
                    execution_mode="sequential",
                    failure_policy="skip_dependents",
                )
            ],
            context={"prompt": prompt},
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
        maybe_final_output = final_state.get("final_output")
        if isinstance(maybe_final_output, Mapping):
            final_output = dict(maybe_final_output)

        loop_terminated_reason = str(loop_output.get("terminated_reason", "max_iterations_reached"))

        if loop_terminated_reason == "iteration_failed":
            terminated_reason = "step_failure"
        elif len(plan_steps) > self._controls.max_iterations:
            terminated_reason = "max_iterations_reached"
        else:
            terminated_reason = "completed"

        plan_execute_result = AgentResult(
            output={
                "plan": parsed_plan,
                "steps_executed": len(step_results),
                "step_results": step_results,
                "final_output": final_output,
                "terminated_reason": terminated_reason,
            },
            success=terminated_reason in {"completed", "max_iterations_reached"}
            and bool(step_results),
            tool_results=all_tool_results,
            model_response=last_model_response,
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
            controls=self._controls,
            budget_metadata=budget_tracker.as_metadata(controls=self._controls),
            extra_metadata={
                "plan": {
                    "step_count": len(plan_steps),
                    "executed_step_count": len(step_results),
                },
                "loop": {
                    "iterations": loop_output.get("iterations", self._controls.max_iterations),
                    "iterations_executed": loop_output.get("iterations_executed", 0),
                    "terminated_reason": loop_terminated_reason,
                },
            },
        )


def _deserialize_tool_results(raw_tool_results: object) -> list[ToolResult]:
    if not isinstance(raw_tool_results, list):
        return []
    parsed_results: list[ToolResult] = []
    for raw_tool_result in raw_tool_results:
        if not isinstance(raw_tool_result, Mapping):
            continue
        try:
            parsed_results.append(ToolResult(**dict(raw_tool_result)))
        except (TypeError, ValueError):
            continue
    return parsed_results


def _deserialize_model_response(raw_model_response: object) -> LLMResponse | None:
    if not isinstance(raw_model_response, Mapping):
        return None
    try:
        return LLMResponse(**dict(raw_model_response))
    except TypeError:
        return None


__all__ = [
    "PlannerExecutorPattern",
]
