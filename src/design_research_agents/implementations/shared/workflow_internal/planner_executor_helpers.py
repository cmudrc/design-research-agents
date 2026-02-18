"""Shared constants/helpers for planner-executor workflow patterns."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolResult, ToolSpec
from design_research_agents.implementations.shared.workflow_internal.loop_state import (
    normalize_mapping,
    normalize_mapping_records,
    parse_loop_iteration,
)
from design_research_agents.implementations.shared.workflow_internal.pattern_runtime import (
    WorkflowBudgetTracker,
    render_prompt_template,
)

PLAN_SCHEMA: dict[str, object] = {
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

DEFAULT_PLANNER_SYSTEM_PROMPT = (
    "You are a planner for a plan-execute runtime. Return strict JSON only with steps[]."
)
DEFAULT_PLANNER_USER_PROMPT_TEMPLATE = (
    "Create an execution plan for this task. "
    "Each step must have step_id, instruction, and success_criteria.\n\n"
    "Task:\n$task_prompt"
)
DEFAULT_EXECUTOR_STEP_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Plan step id: $step_id",
        "Instruction: $instruction",
        "Success criteria: $success_criteria",
        "Prior step outputs:",
        "$prior_step_outputs_json",
    ]
)


def deserialize_tool_results(raw_tool_results: object) -> list[ToolResult]:
    """Deserialize serialized tool result dictionaries from nested workflow output.

    Args:
        raw_tool_results: Raw value expected to be a list of tool-result mappings.

    Returns:
        Parsed tool results that passed structural validation.
    """
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


def deserialize_model_response(raw_model_response: object) -> LLMResponse | None:
    """Deserialize a serialized model response mapping when present.

    Args:
        raw_model_response: Raw value expected to be a model-response mapping.

    Returns:
        Parsed model response, or ``None`` when decoding fails.
    """
    if not isinstance(raw_model_response, Mapping):
        return None
    try:
        return LLMResponse(**dict(raw_model_response))
    except TypeError:
        return None


class PlanExecuteLoopCallbacks:
    """Callback bundle used by planner-executor loop runtime steps."""

    def __init__(
        self,
        *,
        prompt: str,
        plan_steps: list[dict[str, object]],
        executor_step_prompt_template: str,
        request_id: str,
        dependencies: Mapping[str, object],
        budget_tracker: WorkflowBudgetTracker,
        runtime_tool_specs: Mapping[str, ToolSpec],
        initial_model_response: LLMResponse | None,
    ) -> None:
        """Store shared state for loop callback methods.

        Args:
            prompt: Task prompt for the run.
            plan_steps: Parsed plan step mappings.
            executor_step_prompt_template: Prompt template for executor steps.
            request_id: Resolved request identifier.
            dependencies: Normalized dependency payload mapping.
            budget_tracker: Budget tracker for model/tool metrics.
            runtime_tool_specs: Runtime tool specs keyed by name.
            initial_model_response: Initial planner response.
        """
        self.prompt = prompt
        self.plan_steps = list(plan_steps)
        self.executor_step_prompt_template = executor_step_prompt_template
        self.request_id = request_id
        self.dependencies = dependencies
        self.budget_tracker = budget_tracker
        self.runtime_tool_specs = runtime_tool_specs
        self.last_model_response = initial_model_response
        self.all_tool_results: list[ToolResult] = []

    def continue_predicate(self, iteration: int, state: Mapping[str, object]) -> bool:
        """Continue while there are remaining planned steps.

        Args:
            iteration: One-based loop iteration index.
            state: Current loop state before this iteration.

        Returns:
            ``True`` when another planned step should execute.
        """
        del state
        return iteration <= len(self.plan_steps)

    def executor_prompt_builder(self, step_context: Mapping[str, object]) -> str:
        """Build executor prompt text for the current planned step.

        Args:
            step_context: Step context containing loop metadata and state.

        Returns:
            Rendered prompt text for ``SingleStepCodeToolCallingAgent``.

        Raises:
            ValueError: If loop metadata or plan step payload is invalid.
        """
        loop_metadata = step_context.get("_loop")
        if not isinstance(loop_metadata, Mapping):
            raise ValueError("Loop metadata is required for plan execution prompt building.")

        raw_iteration = loop_metadata.get("iteration")
        iteration = parse_loop_iteration(
            raw_iteration,
            error_prefix="Plan execute loop iteration",
        )

        if iteration < 1 or iteration > len(self.plan_steps):
            raise ValueError(f"Loop iteration {iteration} is out of bounds for planned steps.")

        raw_step = self.plan_steps[iteration - 1]
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"Plan step at iteration {iteration} is not a mapping.")

        loop_state = step_context.get("loop_state")
        loop_step_records = (
            normalize_mapping_records(loop_state.get("step_results"))
            if isinstance(loop_state, Mapping)
            else []
        )

        return render_prompt_template(
            template_text=self.executor_step_prompt_template,
            variables={
                "task_prompt": self.prompt,
                "step_id": str(raw_step.get("step_id", f"step_{iteration}")),
                "instruction": str(raw_step.get("instruction", "")),
                "success_criteria": str(raw_step.get("success_criteria", "")),
                "prior_step_outputs_json": json.dumps(loop_step_records[-3:], sort_keys=True),
            },
            field_name="plan_execute_executor_step_prompt_template",
        )

    def state_reducer(
        self,
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        """Fold one loop iteration result into aggregate plan-execution state.

        Args:
            state: Current aggregate loop state.
            iteration_result: Workflow result produced by this iteration body.
            iteration: One-based iteration index.

        Returns:
            Updated loop state with step result history and latest final output.
        """
        raw_step = self.plan_steps[iteration - 1]
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

        step_tool_results = deserialize_tool_results(serialized_agent_result.get("tool_results"))
        step_model_response = deserialize_model_response(
            serialized_agent_result.get("model_response")
        )
        self.budget_tracker.add_model_response(step_model_response)
        self.budget_tracker.add_tool_results(
            tool_results=step_tool_results,
            tool_specs=self.runtime_tool_specs,
        )
        if step_model_response is not None:
            self.last_model_response = step_model_response
        self.all_tool_results.extend(step_tool_results)

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


__all__ = [
    "DEFAULT_EXECUTOR_STEP_PROMPT_TEMPLATE",
    "DEFAULT_PLANNER_SYSTEM_PROMPT",
    "DEFAULT_PLANNER_USER_PROMPT_TEMPLATE",
    "PLAN_SCHEMA",
    "PlanExecuteLoopCallbacks",
    "deserialize_model_response",
    "deserialize_tool_results",
]
