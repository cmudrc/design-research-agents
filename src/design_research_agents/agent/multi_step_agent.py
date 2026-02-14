"""Multi-step ReAct-style agent built as a loop over SingleStepCodeAgent."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent.single_step_code_agent import SingleStepCodeAgent
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime


class MultiStepAgent(Agent):
    """Agent that iterates action-observation steps until continuation stops."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        model: str = "gpt-4o-mini",
        max_steps: int = 5,
        max_tool_calls_per_step: int = 5,
        execution_timeout_seconds_per_step: int = 5,
        validate_tool_input_schema: bool = False,
        stop_on_step_failure: bool = True,
    ) -> None:
        """Initialize a multi-step agent.

        Args:
            llm_client: LLM client used for continuation and action generation.
            tool_runtime: Tool runtime shared across all steps.
            model: Model name used for LLM calls.
            max_steps: Maximum number of action-observation iterations.
            max_tool_calls_per_step: Tool-call limit applied to each action step.
            execution_timeout_seconds_per_step: Code execution timeout for each action step.
            validate_tool_input_schema: Whether to validate tool input schemas on each step.
            stop_on_step_failure: Whether to stop immediately when one step fails.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if max_tool_calls_per_step < 1:
            raise ValueError("max_tool_calls_per_step must be >= 1.")
        if execution_timeout_seconds_per_step < 1:
            raise ValueError("execution_timeout_seconds_per_step must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._model = model
        self._max_steps = max_steps
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._execution_timeout_seconds_per_step = execution_timeout_seconds_per_step
        self._validate_tool_input_schema = validate_tool_input_schema
        self._stop_on_step_failure = stop_on_step_failure

    def run(self, input: Mapping[str, object], context: Mapping[str, object]) -> AgentResult:
        """Run an action-observation loop over SingleStepCodeAgent."""
        prompt = _extract_prompt(input)
        max_steps = _extract_positive_int(
            input_payload=input,
            key="max_steps",
            default_value=self._max_steps,
        )
        max_tool_calls_per_step = _extract_positive_int(
            input_payload=input,
            key="max_tool_calls_per_step",
            default_value=self._max_tool_calls_per_step,
        )
        execution_timeout_seconds_per_step = _extract_positive_int(
            input_payload=input,
            key="execution_timeout_seconds_per_step",
            default_value=self._execution_timeout_seconds_per_step,
        )
        validate_tool_input_schema = _extract_boolean(
            input_payload=input,
            key="validate_tool_input_schema",
            default_value=self._validate_tool_input_schema,
        )
        stop_on_step_failure = _extract_boolean(
            input_payload=input,
            key="stop_on_step_failure",
            default_value=self._stop_on_step_failure,
        )

        step_agent = SingleStepCodeAgent(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            model=self._model,
            max_tool_calls=max_tool_calls_per_step,
            execution_timeout_seconds=execution_timeout_seconds_per_step,
            validate_tool_input_schema=validate_tool_input_schema,
        )

        memory: list[dict[str, object]] = [{"kind": "task", "prompt": prompt}]
        continuation_trace: list[dict[str, object]] = []
        step_outputs: list[dict[str, object]] = []
        tool_results: list[ToolResult] = []
        final_output: dict[str, object] = {}
        last_model_response: LLMResponse | None = None
        terminated_reason = "max_steps_reached"

        for step_index in range(max_steps):
            should_continue, continue_reason, continue_source, continue_response = (
                self._llm_should_continue(
                    prompt=prompt,
                    memory=memory,
                    step_index=step_index,
                    max_steps=max_steps,
                )
            )
            if continue_response is not None:
                last_model_response = continue_response
            continuation_trace.append(
                {
                    "step": step_index + 1,
                    "continue": should_continue,
                    "reason": continue_reason,
                    "source": continue_source,
                }
            )
            if not should_continue:
                terminated_reason = f"continuation_stopped:{continue_source}"
                break

            step_prompt = _build_step_prompt(
                prompt=prompt,
                memory=memory,
                step_number=step_index + 1,
            )
            step_input = dict(input)
            step_input["prompt"] = step_prompt
            step_input["max_tool_calls"] = max_tool_calls_per_step
            step_input["execution_timeout_seconds"] = execution_timeout_seconds_per_step
            step_input["validate_tool_input_schema"] = validate_tool_input_schema

            step_context = dict(context)
            step_context["multi_step_memory"] = list(memory)
            step_context["step_index"] = step_index + 1
            step_result = step_agent.run(step_input, step_context)
            if step_result.model_response is not None:
                last_model_response = step_result.model_response

            tool_results.extend(step_result.tool_results)
            step_output = {
                "step": step_index + 1,
                "success": step_result.success,
                "final_output": step_result.output.get("final_output", {}),
                "error": step_result.output.get("error"),
                "tool_results_count": len(step_result.tool_results),
            }
            step_outputs.append(step_output)
            memory.extend(
                [
                    {
                        "kind": "action",
                        "step": step_index + 1,
                        "generated_code": step_result.output.get("generated_code", ""),
                    },
                    {
                        "kind": "observation",
                        "step": step_index + 1,
                        "success": step_result.success,
                        "final_output": step_result.output.get("final_output", {}),
                        "error": step_result.output.get("error"),
                    },
                ]
            )

            if step_result.success:
                raw_final_output = step_result.output.get("final_output")
                if isinstance(raw_final_output, Mapping):
                    final_output = dict(raw_final_output)
                continue

            terminated_reason = "step_failure"
            if stop_on_step_failure:
                return _failure_result(
                    error=str(step_result.output.get("error", "Step execution failed.")),
                    model_response=last_model_response,
                    tool_results=tool_results,
                    context=context,
                    metadata={
                        "stage": "step_execution",
                        "terminated_reason": terminated_reason,
                        "continuation": continuation_trace,
                    },
                    output={
                        "final_output": final_output,
                        "steps_executed": len(step_outputs),
                        "step_outputs": step_outputs,
                        "memory": memory,
                        "terminated_reason": terminated_reason,
                    },
                )

        success = all(step_output["success"] is True for step_output in step_outputs)
        output: dict[str, object] = {
            "final_output": final_output,
            "steps_executed": len(step_outputs),
            "step_outputs": step_outputs,
            "memory": memory,
            "terminated_reason": terminated_reason,
        }
        return AgentResult(
            output=output,
            success=success,
            tool_results=tool_results,
            model_response=last_model_response,
            metadata={
                "context_keys": sorted(context.keys()),
                "continuation": continuation_trace,
                "config": {
                    "max_steps": max_steps,
                    "max_tool_calls_per_step": max_tool_calls_per_step,
                    "execution_timeout_seconds_per_step": execution_timeout_seconds_per_step,
                    "validate_tool_input_schema": validate_tool_input_schema,
                    "stop_on_step_failure": stop_on_step_failure,
                },
            },
        )

    def run_stream(
        self,
        input: Mapping[str, object],
        context: Mapping[str, object],
    ) -> Iterator[AgentStreamEvent]:
        """Stream a deterministic event pair around ``run``."""
        result = self.run(input, context)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)

    def _llm_should_continue(
        self,
        *,
        prompt: str,
        memory: Sequence[Mapping[str, object]],
        step_index: int,
        max_steps: int,
    ) -> tuple[bool, str, str, LLMResponse | None]:
        """Ask the model whether the loop should continue for the next step."""
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are the continuation controller for a multi-step tool-using agent. "
                    "Decide whether another action-observation step is needed. "
                    "Policy: before any observation exists, continue must be true. "
                    "Stop only when memory already contains enough successful observations "
                    "to satisfy the task. Return strict JSON only."
                ),
            ),
            LLMMessage(
                role="user",
                content=_build_continue_prompt(
                    prompt=prompt,
                    memory=memory,
                    step_number=step_index + 1,
                ),
            ),
        ]
        llm_params = LLMChatParams(
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["continue"],
                "properties": {
                    "continue": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
            provider_options={"agent": "MultiStepAgent", "phase": "continuation"},
        )
        response = self._llm_client.chat(messages, model=self._model, params=llm_params)
        parsed = _parse_json_mapping(response.text)
        if parsed is not None and isinstance(parsed.get("continue"), bool):
            # Ensure at least one action-observation cycle runs before stopping.
            if step_index == 0 and not bool(parsed["continue"]) and not _has_observation(memory):
                return True, "first-step guardrail", "guardrail", response
            reason = parsed.get("reason")
            normalized_reason = str(reason) if reason is not None else "model decision"
            return bool(parsed["continue"]), normalized_reason, "model", response

        fallback_decision = _fallback_should_continue(
            memory=memory,
            step_index=step_index,
            max_steps=max_steps,
        )
        return fallback_decision, "fallback heuristic", "fallback", response


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from the input payload."""
    raw_prompt = input_payload.get(
        "prompt", input_payload.get("text", "Provide a concise response.")
    )
    return str(raw_prompt)


def _extract_positive_int(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: int,
) -> int:
    """Extract positive integer option from input payload."""
    raw_value = input_payload.get(key)
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return default_value
    if isinstance(raw_value, int) and raw_value >= 1:
        return raw_value
    return default_value


def _extract_boolean(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: bool,
) -> bool:
    """Extract boolean option from input payload."""
    raw_value = input_payload.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    return default_value


def _build_continue_prompt(
    *,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_number: int,
) -> str:
    """Build continuation-decision prompt from task and memory."""
    memory_preview = json.dumps(list(memory)[-6:], sort_keys=True)
    return "\n".join(
        [
            "Decide whether the agent should continue to another action step.",
            'Return JSON only: {"continue": <bool>, "reason": "..."}.',
            "Decision policy:",
            "- If no observation exists yet, set continue=true.",
            "- If latest observation failed, set continue=false.",
            "- If task is fully satisfied from memory set continue=false; otherwise continue=true.",
            "",
            f"Step number: {step_number}",
            f"Task: {prompt}",
            f"Memory tail: {memory_preview}",
        ]
    )


def _build_step_prompt(
    *,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_number: int,
) -> str:
    """Build one action prompt for the current step."""
    memory_preview = json.dumps(list(memory)[-8:], sort_keys=True)
    return "\n".join(
        [
            f"Task: {prompt}",
            f"Current step: {step_number}",
            "Use memory to decide next action.",
            f"Memory tail: {memory_preview}",
        ]
    )


def _parse_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Parse first JSON object from text."""
    parsed_direct = _load_json_mapping(raw_text)
    if parsed_direct is not None:
        return parsed_direct

    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _load_json_mapping(raw_text: str) -> dict[str, object] | None:
    """Load text as a JSON mapping, if valid."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    return dict(parsed)


def _fallback_should_continue(
    *,
    memory: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
) -> bool:
    """Fallback continuation policy when model output is not parseable JSON."""
    if step_index >= max_steps:
        return False

    # On parse failure, guarantee one first step, then stop by default.
    if step_index == 0:
        return True

    # If the last observation failed, stop.
    for entry in reversed(memory):
        if entry.get("kind") != "observation":
            continue
        if entry.get("success") is False:
            return False
        break

    return False


def _has_observation(memory: Sequence[Mapping[str, object]]) -> bool:
    """Return whether memory includes at least one observation entry."""
    return any(entry.get("kind") == "observation" for entry in memory)


def _failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: Sequence[ToolResult],
    context: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> AgentResult:
    """Build a structured failure result."""
    return AgentResult(
        output={"error": error, **dict(output)},
        success=False,
        tool_results=list(tool_results),
        model_response=model_response,
        metadata={"context_keys": sorted(context.keys()), **dict(metadata)},
    )
