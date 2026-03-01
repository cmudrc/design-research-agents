"""Multi-step direct-LLM controller with iterative CONTINUE/STOP decisions.

This agent does not invoke external tools. Instead, each controller step emits
an internal action:
- ``CONTINUE`` with refined partial progress, or
- ``STOP`` with the final response text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents._contracts._termination import (
    SOURCE_MODEL,
    TERMINATED_CONTROLLER_INVALID_PAYLOAD,
    TERMINATED_MAX_STEPS_REACHED,
    stop_reason,
)
from design_research_agents._tracing import (
    Tracer,
    emit_continuation_decision,
    emit_guardrail_decision,
    finish_model_call,
    start_model_call,
)
from design_research_agents.workflow import CompiledExecution

from .._execution_context import (
    resolve_agent_execution_context,
)
from .._input_parsing import (
    extract_positive_int as _extract_positive_int,
)
from .._input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from .._model_resolution import (
    resolve_agent_model,
)
from .._multi_step_common import (
    build_step_prompt,
)
from .._prompt_overrides import (
    resolve_prompt_text,
)
from .._response_schemas import (
    build_multi_step_direct_controller_response_schema,
    clone_response_schema,
)
from .._result_builders import (
    build_failure_result,
)
from .._workflow_loop_orchestration import (
    compile_workflow_loop,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class _ControllerDecision:
    """One parsed controller action for a direct-response step."""

    decision: str
    """Normalized controller decision (``CONTINUE`` or ``STOP``)."""
    content: str
    """Controller content for intermediate reasoning output."""
    final_output: str | None
    """Optional explicit final output payload when stopping."""
    reason: str
    """Decision rationale used for tracing and memory entries."""
    source: str
    """Decision source label."""


class MultiStepDirectLLMAgent(Delegate):
    """Agent that iterates internal direct-response controller decisions."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        max_steps: int = 5,
        controller_system_prompt: str | None = None,
        controller_user_prompt_template: str | None = None,
        step_memory_tail_items: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a multi-step direct-response controller agent.

        Args:
            llm_client: LLM client used for each controller step.
            max_steps: Maximum number of controller steps.
            controller_system_prompt: Optional controller system prompt override.
            controller_user_prompt_template: Optional controller user prompt template override.
            step_memory_tail_items: Memory tail size rendered into each controller step prompt.
            tracer: Optional explicit tracer dependency.

        Raises:
            ValueError: Raised when constructor bounds are invalid.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if step_memory_tail_items < 1:
            raise ValueError("step_memory_tail_items must be >= 1.")

        self._llm_client = llm_client
        self._tracer = tracer
        self.workflow: object | None = None
        self._max_steps = max_steps
        self._controller_system_prompt = resolve_prompt_text(
            override=controller_system_prompt,
            default_prompt_name="multi_step_direct_controller_system",
            field_name="controller_system_prompt",
        )
        self._controller_user_prompt_template = resolve_prompt_text(
            override=controller_user_prompt_template,
            default_prompt_name="multi_step_direct_controller_user",
            field_name="controller_user_prompt_template",
        )
        self._step_memory_tail_items = step_memory_tail_items
        self._controller_response_schema = build_multi_step_direct_controller_response_schema()

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Iterative CONTINUE/STOP controller steps until termination."""
        return self.compile(
            prompt=prompt,
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
        """Compile iterative CONTINUE/STOP controller steps into one loop workflow."""
        execution_context = resolve_agent_execution_context(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        resolved_request_id = execution_context.request_id
        resolved_dependencies = execution_context.dependencies
        prompt = execution_context.prompt
        max_steps = _extract_positive_int(
            input_payload=execution_context.normalized_input,
            key="max_steps",
            default_value=self._max_steps,
        )
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        initial_state: dict[str, object] = {
            "memory": [{"kind": "task", "prompt": prompt}],
            "step_outputs": [],
            "final_output": "",
            "terminated_reason": TERMINATED_MAX_STEPS_REACHED,
            "last_model_response": None,
            "should_continue": True,
            "fatal_error": None,
            "fatal_metadata": {},
        }

        def _continue_predicate(iteration: int, state: Mapping[str, object]) -> bool:
            """Return whether another controller iteration should execute.

            Args:
                iteration: One-based loop iteration number.
                state: Current loop-state mapping.

            Returns:
                ``True`` when the controller should execute another step.
            """
            del iteration
            return bool(state.get("should_continue", True))

        def _run_iteration(iteration: int, state: Mapping[str, object]) -> Mapping[str, object]:
            """Execute one controller iteration and return next loop state.

            Args:
                iteration: One-based loop iteration number.
                state: Current loop-state mapping.

            Returns:
                Next loop-state mapping.

            Raises:
                Exception: Propagates model call failures.
            """
            step_number = iteration
            memory = _coerce_state_records(state.get("memory"))
            step_outputs = _coerce_state_records(state.get("step_outputs"))
            final_output = str(state.get("final_output", ""))
            last_model_response = state.get("last_model_response")

            user_prompt = build_step_prompt(
                prompt=prompt,
                memory=memory,
                step_number=step_number,
                prompt_template=self._controller_user_prompt_template,
                memory_tail_items=self._step_memory_tail_items,
            )
            messages = [
                LLMMessage(role="system", content=self._controller_system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            llm_params = LLMChatParams(
                response_schema=clone_response_schema(self._controller_response_schema),
                provider_options={
                    "agent": "MultiStepDirectLLMAgent",
                    "phase": "controller_step",
                },
            )
            model_span_id = start_model_call(
                model=resolved_model,
                messages=messages,
                params=llm_params,
                metadata={
                    "agent": "MultiStepDirectLLMAgent",
                    "phase": "controller_step",
                    "step_id": f"controller_step_{step_number}",
                },
            )
            try:
                llm_response = self._llm_client.chat(
                    messages,
                    model=resolved_model,
                    params=llm_params,
                )
            except Exception as exc:
                finish_model_call(model_span_id, error=str(exc), model=resolved_model)
                raise
            finish_model_call(model_span_id, response=llm_response)
            last_model_response = llm_response

            parsed_decision = _parse_controller_decision(llm_response.text)
            if parsed_decision is None:
                emit_guardrail_decision(
                    guardrail="direct_controller_output",
                    decision="reject",
                    reason="invalid controller JSON",
                    details={"step": step_number},
                )
                return {
                    "memory": memory,
                    "step_outputs": step_outputs,
                    "final_output": final_output,
                    "terminated_reason": TERMINATED_CONTROLLER_INVALID_PAYLOAD,
                    "last_model_response": last_model_response,
                    "should_continue": False,
                    "fatal_error": ("Controller output was invalid. Expected JSON with `decision` and `content`."),
                    "fatal_metadata": {
                        "stage": "controller_step",
                        "terminated_reason": TERMINATED_CONTROLLER_INVALID_PAYLOAD,
                    },
                }

            if parsed_decision.decision == "CONTINUE":
                final_output = parsed_decision.content
                emit_continuation_decision(
                    step=step_number,
                    should_continue=True,
                    reason=parsed_decision.reason,
                    source=parsed_decision.source,
                )
                step_outputs.append(
                    {
                        "step": step_number,
                        "decision": "CONTINUE",
                        "content": parsed_decision.content,
                        "reason": parsed_decision.reason,
                        "source": parsed_decision.source,
                    }
                )
                memory.append(
                    {
                        "kind": "continue",
                        "step": step_number,
                        "content": parsed_decision.content,
                        "reason": parsed_decision.reason,
                        "source": parsed_decision.source,
                    }
                )
                return {
                    "memory": memory,
                    "step_outputs": step_outputs,
                    "final_output": final_output,
                    "terminated_reason": TERMINATED_MAX_STEPS_REACHED,
                    "last_model_response": last_model_response,
                    "should_continue": True,
                    "fatal_error": None,
                    "fatal_metadata": {},
                }

            final_output = parsed_decision.final_output or parsed_decision.content
            terminated_reason = stop_reason(parsed_decision.source)
            emit_continuation_decision(
                step=step_number,
                should_continue=False,
                reason=parsed_decision.reason,
                source=parsed_decision.source,
            )
            step_outputs.append(
                {
                    "step": step_number,
                    "decision": "STOP",
                    "final_output": final_output,
                    "reason": parsed_decision.reason,
                    "source": parsed_decision.source,
                }
            )
            memory.append(
                {
                    "kind": "stop",
                    "step": step_number,
                    "final_output": final_output,
                    "reason": parsed_decision.reason,
                    "source": parsed_decision.source,
                }
            )
            return {
                "memory": memory,
                "step_outputs": step_outputs,
                "final_output": final_output,
                "terminated_reason": terminated_reason,
                "last_model_response": last_model_response,
                "should_continue": False,
                "fatal_error": None,
                "fatal_metadata": {},
            }

        workflow = compile_workflow_loop(
            max_iterations=max_steps,
            initial_state=initial_state,
            continue_predicate=_continue_predicate,
            iteration_handler=_run_iteration,
            tracer=self._tracer,
        )
        self.workflow = workflow

        def _finalize(workflow_result: ExecutionResult) -> ExecutionResult:
            loop_step_result = workflow_result.step_results.get("agent_loop")
            loop_output = loop_step_result.output if loop_step_result is not None else {}
            final_state_raw = loop_output.get("final_state", {})
            final_state = dict(final_state_raw) if isinstance(final_state_raw, Mapping) else {}
            workflow_payload = workflow_result.to_dict()
            workflow_artifacts = workflow_result.output.get("artifacts", [])
            step_outputs = _coerce_state_records(final_state.get("step_outputs"))
            memory = _coerce_state_records(final_state.get("memory"))
            terminated_reason = str(final_state.get("terminated_reason", TERMINATED_MAX_STEPS_REACHED))
            final_output = str(final_state.get("final_output", ""))
            maybe_model_response = final_state.get("last_model_response")
            last_model_response = maybe_model_response if isinstance(maybe_model_response, LLMResponse) else None
            fatal_error = final_state.get("fatal_error")
            fatal_metadata_raw = final_state.get("fatal_metadata")
            fatal_metadata = dict(fatal_metadata_raw) if isinstance(fatal_metadata_raw, Mapping) else {}

            if isinstance(fatal_error, str) and fatal_error:
                return build_failure_result(
                    error=fatal_error,
                    model_response=last_model_response,
                    tool_results=[],
                    request_id=resolved_request_id,
                    dependencies=resolved_dependencies,
                    metadata=fatal_metadata,
                    output={
                        "final_output": final_output,
                        "steps_executed": len(step_outputs),
                        "step_outputs": step_outputs,
                        "memory": memory,
                        "terminated_reason": terminated_reason,
                        "workflow": workflow_payload,
                        "artifacts": workflow_artifacts,
                    },
                )

            return ExecutionResult(
                output={
                    "final_output": final_output,
                    "steps_executed": len(step_outputs),
                    "step_outputs": step_outputs,
                    "memory": memory,
                    "terminated_reason": terminated_reason,
                    "workflow": workflow_payload,
                    "artifacts": workflow_artifacts,
                },
                success=True,
                tool_results=[],
                model_response=last_model_response,
                metadata={
                    "request_id": resolved_request_id,
                    "dependency_keys": sorted(resolved_dependencies.keys()),
                    "controller_steps": list(step_outputs),
                    "config": {
                        "max_steps": max_steps,
                        "step_memory_tail_items": self._step_memory_tail_items,
                    },
                },
            )

        return CompiledExecution(
            workflow=workflow,
            input={},
            request_id=resolved_request_id,
            workflow_request_id=f"{resolved_request_id}:workflow_loop",
            dependencies=resolved_dependencies,
            delegate_name="MultiStepDirectLLMAgent",
            tracer=self._tracer,
            trace_input=execution_context.normalized_input,
            finalize=_finalize,
        )


def _coerce_state_records(raw_records: object) -> list[dict[str, object]]:
    """Coerce raw controller records into normalized mapping lists.

    Args:
        raw_records: Raw record payload.

    Returns:
        Normalized list of record mappings.
    """
    if not isinstance(raw_records, list):
        return []
    return [dict(record) for record in raw_records if isinstance(record, Mapping)]


def _parse_controller_decision(raw_text: str) -> _ControllerDecision | None:
    """Parse one controller decision from raw model output text.

    Args:
        raw_text: Raw model output produced by the controller step.

    Returns:
        Parsed controller decision, or ``None`` when parsing fails.
    """
    parsed = _parse_json_mapping(raw_text)
    if parsed is None:
        return None

    raw_decision = parsed.get("decision")
    if not isinstance(raw_decision, str):
        return None

    normalized_decision = raw_decision.strip().upper()
    if normalized_decision not in {"CONTINUE", "STOP"}:
        return None

    content = str(parsed.get("content", "")).strip()
    final_output = parsed.get("final_output")
    return _ControllerDecision(
        decision=normalized_decision,
        content=content,
        final_output=(str(final_output).strip() if final_output is not None else None),
        reason=str(parsed.get("reason", content or "model decision")),
        source=SOURCE_MODEL,
    )
