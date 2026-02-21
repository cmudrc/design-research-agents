"""Reusable two-speaker conversation pattern with optional distinct LLM clients."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import LLMClient, LLMResponse
from design_research_agents.contracts.workflow import (
    DelegateBatchCall,
    DelegateBatchStep,
    LogicStep,
    LoopStep,
    WorkflowDelegate,
)
from design_research_agents.implementations.agents.direct_llm_call import (
    DirectLLMCall,
)
from design_research_agents.implementations.shared.agent_internal.model_resolution import (
    resolve_agent_model,
)
from design_research_agents.implementations.shared.agent_internal.result_builders import (
    build_failure_result,
)
from design_research_agents.implementations.shared.workflow_internal import (
    execute_pattern_with_trace,
    normalize_mapping,
    normalize_mapping_records,
    normalize_request_id_prefix,
    render_prompt_template,
    resolve_pattern_run_context,
    resolve_prompt_override,
)
from design_research_agents.tracing import Tracer
from design_research_agents.workflow import Workflow

_DEFAULT_SPEAKER_A_NAME = "speaker_a"
_DEFAULT_SPEAKER_B_NAME = "speaker_b"

_DEFAULT_SPEAKER_A_SYSTEM_PROMPT = (
    "You are speaker A in a two-model conversation. "
    "Build on the partner message and move the task forward."
)
_DEFAULT_SPEAKER_B_SYSTEM_PROMPT = (
    "You are speaker B in a two-model conversation. "
    "Respond directly to speaker A and add concrete next-step reasoning."
)
_DEFAULT_SPEAKER_A_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Turn: $turn",
        "You are $speaker_name and your partner is $partner_name.",
        "Conversation transcript so far (JSON):",
        "$conversation_transcript_json",
        "Latest message from $partner_name:",
        "$partner_message",
        "Respond with your next message only.",
    ]
)
_DEFAULT_SPEAKER_B_USER_PROMPT_TEMPLATE = "\n".join(
    [
        "Task: $task_prompt",
        "Turn: $turn",
        "You are $speaker_name and your partner is $partner_name.",
        "Conversation transcript so far (JSON):",
        "$conversation_transcript_json",
        "Latest message from $partner_name:",
        "$partner_message",
        "Respond with your next message only.",
    ]
)


class _ConversationLoopCallbacks:
    """Loop callback bundle used by the conversation pattern."""

    def __init__(
        self,
        *,
        pattern: ConversationPattern,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        speaker_a_delegate: WorkflowDelegate,
        speaker_b_delegate: WorkflowDelegate,
        runtime_state: dict[str, object],
    ) -> None:
        """Store callback dependencies.

        Args:
            pattern: Parent conversation pattern.
            prompt: Task prompt for the run.
            request_id: Resolved request id.
            dependencies: Resolved dependency mapping.
            speaker_a_delegate: Configured delegate for speaker A.
            speaker_b_delegate: Configured delegate for speaker B.
            runtime_state: Mutable callback state sink.
        """
        self._pattern = pattern
        self._prompt = prompt
        self._request_id = request_id
        self._dependencies = dependencies
        self._speaker_a_delegate = speaker_a_delegate
        self._speaker_b_delegate = speaker_b_delegate
        self._runtime_state = runtime_state

    def continue_predicate(self, iteration: int, state: Mapping[str, object]) -> bool:
        """Return whether another turn should run.

        Args:
            iteration: One-based iteration index.
            state: Current loop state mapping.

        Returns:
            ``True`` when the loop should continue.
        """
        del iteration
        return bool(state.get("should_continue", True))

    def build_speaker_a_calls(
        self,
        context: Mapping[str, object],
    ) -> list[DelegateBatchCall]:
        """Build the speaker-A delegate call for one turn."""
        turn_number, transcript, _last_a, last_b = _resolve_turn_context(context)
        speaker_a_prompt = _render_conversation_prompt(
            template_text=self._pattern._speaker_a_user_prompt_template,
            field_name="speaker_a_user_prompt_template",
            task_prompt=self._prompt,
            turn_number=turn_number,
            speaker_name=self._pattern._speaker_a_name,
            partner_name=self._pattern._speaker_b_name,
            partner_message=last_b,
            transcript=transcript,
        )
        return [
            DelegateBatchCall(
                call_id="speaker_a",
                delegate=self._speaker_a_delegate,
                prompt=speaker_a_prompt,
            )
        ]

    def build_speaker_b_calls(
        self,
        context: Mapping[str, object],
    ) -> list[DelegateBatchCall]:
        """Build the speaker-B delegate call for one turn."""
        turn_number, transcript, _last_a, _last_b = _resolve_turn_context(context)
        speaker_a_result = _extract_delegate_batch_call_result(
            context=context,
            dependency_step_id="conversation_speaker_a_batch",
            call_id="speaker_a",
        )
        speaker_a_output = _extract_call_output(speaker_a_result)
        speaker_a_message = _extract_model_text_from_output(speaker_a_output)
        speaker_b_prompt = _render_conversation_prompt(
            template_text=self._pattern._speaker_b_user_prompt_template,
            field_name="speaker_b_user_prompt_template",
            task_prompt=self._prompt,
            turn_number=turn_number,
            speaker_name=self._pattern._speaker_b_name,
            partner_name=self._pattern._speaker_a_name,
            partner_message=speaker_a_message or "(none)",
            transcript=transcript,
        )
        return [
            DelegateBatchCall(
                call_id="speaker_b",
                delegate=self._speaker_b_delegate,
                prompt=speaker_b_prompt,
            )
        ]

    def build_turn_state(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Build loop state for the next turn from delegate-batch outputs.

        Args:
            context: Workflow step context.

        Returns:
            Updated loop state for the next iteration.
        """
        (
            turn_number,
            transcript,
            last_message_from_a,
            last_message_from_b,
        ) = _resolve_turn_context(context)
        speaker_a_result = _extract_delegate_batch_call_result(
            context=context,
            dependency_step_id="conversation_speaker_a_batch",
            call_id="speaker_a",
        )
        speaker_a_response = _extract_call_model_response(speaker_a_result)
        if speaker_a_response is not None:
            self._runtime_state["last_model_response"] = speaker_a_response
        if not _is_call_success(speaker_a_result):
            return _build_loop_failure_state(
                transcript=transcript,
                last_message_from_a=last_message_from_a,
                last_message_from_b=last_message_from_b,
                failure_reason="speaker_a_failed",
                failure_error=_extract_call_error(
                    speaker_a_result,
                    fallback_message="Speaker A failed during conversation turn.",
                ),
            )
        speaker_a_output = _extract_call_output(speaker_a_result)
        speaker_a_message = _extract_model_text_from_output(speaker_a_output)
        transcript.append(
            {
                "turn": turn_number,
                "speaker": self._pattern._speaker_a_name,
                "message": speaker_a_message,
            }
        )
        speaker_b_result = _extract_delegate_batch_call_result(
            context=context,
            dependency_step_id="conversation_speaker_b_batch",
            call_id="speaker_b",
        )
        speaker_b_response = _extract_call_model_response(speaker_b_result)
        if speaker_b_response is not None:
            self._runtime_state["last_model_response"] = speaker_b_response
        if not _is_call_success(speaker_b_result):
            return _build_loop_failure_state(
                transcript=transcript,
                last_message_from_a=speaker_a_message or "(none)",
                last_message_from_b=last_message_from_b,
                failure_reason="speaker_b_failed",
                failure_error=_extract_call_error(
                    speaker_b_result,
                    fallback_message="Speaker B failed during conversation turn.",
                ),
            )
        speaker_b_output = _extract_call_output(speaker_b_result)
        speaker_b_message = _extract_model_text_from_output(speaker_b_output)
        transcript.append(
            {
                "turn": turn_number,
                "speaker": self._pattern._speaker_b_name,
                "message": speaker_b_message,
            }
        )
        return {
            "transcript": transcript,
            "last_message_from_a": speaker_a_message or "(none)",
            "last_message_from_b": speaker_b_message or "(none)",
            "failure_reason": None,
            "failure_error": None,
            "should_continue": True,
        }

    @staticmethod
    def state_reducer(
        state: Mapping[str, object],
        iteration_result: ExecutionResult,
        iteration: int,
    ) -> Mapping[str, object]:
        """Fold one turn output into loop state.

        Args:
            state: Current loop state mapping.
            iteration_result: Workflow result for one loop iteration.
            iteration: One-based loop iteration index.

        Returns:
            Reduced loop state mapping.
        """
        del iteration
        iteration_step = iteration_result.step_results.get("conversation_turn")
        if iteration_step is None or not getattr(iteration_step, "success", False):
            return dict(state)
        output = getattr(iteration_step, "output", {})
        return dict(output) if isinstance(output, Mapping) else dict(state)


class ConversationPattern(Agent):
    """Two-speaker LLM conversation pattern with per-speaker prompts and clients."""

    def __init__(
        self,
        *,
        llm_client_a: LLMClient,
        llm_client_b: LLMClient | None = None,
        speaker_a_delegate: WorkflowDelegate | None = None,
        speaker_b_delegate: WorkflowDelegate | None = None,
        max_turns: int = 3,
        speaker_a_name: str = _DEFAULT_SPEAKER_A_NAME,
        speaker_b_name: str = _DEFAULT_SPEAKER_B_NAME,
        speaker_a_system_prompt: str | None = None,
        speaker_a_user_prompt_template: str | None = None,
        speaker_b_system_prompt: str | None = None,
        speaker_b_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and prompt defaults for conversation orchestration.

        Args:
            llm_client_a: LLM client used by speaker A.
            llm_client_b: Optional LLM client used by speaker B.
                Defaults to ``llm_client_a`` when omitted.
            speaker_a_delegate: Optional explicit delegate for speaker A.
            speaker_b_delegate: Optional explicit delegate for speaker B.
            max_turns: Maximum conversation turns where each turn is A->B.
            speaker_a_name: Display name for speaker A in transcript and prompts.
            speaker_b_name: Display name for speaker B in transcript and prompts.
            speaker_a_system_prompt: Optional override for speaker A system prompt.
            speaker_a_user_prompt_template: Optional speaker A user template override.
            speaker_b_system_prompt: Optional override for speaker B system prompt.
            speaker_b_user_prompt_template: Optional speaker B user template override.
            default_request_id_prefix: Optional request-id prefix used for auto-generated ids.
            default_dependencies: Default dependency mapping merged into each run.
            tracer: Optional tracer used for pattern and nested agent traces.

        Raises:
            ValueError: Raised when constructor configuration is invalid.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1.")

        self._llm_client_a = llm_client_a
        self._llm_client_b = llm_client_b if llm_client_b is not None else llm_client_a
        self._max_turns = max_turns
        self._speaker_a_name = _normalize_speaker_name(
            speaker_a_name,
            field_name="speaker_a_name",
        )
        self._speaker_b_name = _normalize_speaker_name(
            speaker_b_name,
            field_name="speaker_b_name",
        )
        self._speaker_a_system_prompt = resolve_prompt_override(
            override=speaker_a_system_prompt,
            default_value=_DEFAULT_SPEAKER_A_SYSTEM_PROMPT,
            field_name="speaker_a_system_prompt",
        )
        self._speaker_a_user_prompt_template = resolve_prompt_override(
            override=speaker_a_user_prompt_template,
            default_value=_DEFAULT_SPEAKER_A_USER_PROMPT_TEMPLATE,
            field_name="speaker_a_user_prompt_template",
        )
        self._speaker_b_system_prompt = resolve_prompt_override(
            override=speaker_b_system_prompt,
            default_value=_DEFAULT_SPEAKER_B_SYSTEM_PROMPT,
            field_name="speaker_b_system_prompt",
        )
        self._speaker_b_user_prompt_template = resolve_prompt_override(
            override=speaker_b_user_prompt_template,
            default_value=_DEFAULT_SPEAKER_B_USER_PROMPT_TEMPLATE,
            field_name="speaker_b_user_prompt_template",
        )
        self._speaker_a_delegate = speaker_a_delegate
        self._speaker_b_delegate = speaker_b_delegate
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._tracer = tracer
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one two-speaker conversation run.

        Args:
            prompt: Task prompt shared by both speakers.
            request_id: Optional request id for tracing/correlation.
            dependencies: Optional dependency overrides for this run.

        Returns:
            Final conversation execution result.

        Raises:
            Exception: Propagates loop and nested-agent execution failures.
        """
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=self._default_request_id_prefix,
            default_dependencies=self._default_dependencies,
            request_id=request_id,
            dependencies=dependencies,
        )
        return execute_pattern_with_trace(
            agent_name="ConversationPattern",
            request_id=run_context.request_id,
            input_payload={
                "prompt": prompt,
                "max_turns": self._max_turns,
                "speaker_a_name": self._speaker_a_name,
                "speaker_b_name": self._speaker_b_name,
            },
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            runner=lambda: self._run_conversation(
                prompt=prompt,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _run_conversation(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Execute workflow loop for one conversation run.

        Args:
            prompt: Task prompt shared by both speakers.
            request_id: Resolved request id.
            dependencies: Normalized dependency mapping.

        Returns:
            Final conversation result payload.

        Raises:
            RuntimeError: If the loop step result is missing.
        """
        speaker_a_delegate = self._speaker_a_delegate
        if speaker_a_delegate is None:
            resolve_agent_model(llm_client=self._llm_client_a)
            speaker_a_delegate = DirectLLMCall(
                llm_client=self._llm_client_a,
                system_prompt=self._speaker_a_system_prompt,
                tracer=self._tracer,
            )

        speaker_b_delegate = self._speaker_b_delegate
        if speaker_b_delegate is None:
            if self._llm_client_b is not self._llm_client_a:
                resolve_agent_model(llm_client=self._llm_client_b)
            speaker_b_delegate = DirectLLMCall(
                llm_client=self._llm_client_b,
                system_prompt=self._speaker_b_system_prompt,
                tracer=self._tracer,
            )
        runtime_state: dict[str, object] = {"last_model_response": None}
        callbacks = _ConversationLoopCallbacks(
            pattern=self,
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            speaker_a_delegate=speaker_a_delegate,
            speaker_b_delegate=speaker_b_delegate,
            runtime_state=runtime_state,
        )

        self.workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_mode="schema",
            steps=[
                LoopStep(
                    step_id="conversation_loop",
                    steps=(
                        DelegateBatchStep(
                            step_id="conversation_speaker_a_batch",
                            calls_builder=callbacks.build_speaker_a_calls,
                            fail_fast=True,
                        ),
                        DelegateBatchStep(
                            step_id="conversation_speaker_b_batch",
                            dependencies=("conversation_speaker_a_batch",),
                            calls_builder=callbacks.build_speaker_b_calls,
                            fail_fast=True,
                        ),
                        LogicStep(
                            step_id="conversation_turn",
                            dependencies=(
                                "conversation_speaker_a_batch",
                                "conversation_speaker_b_batch",
                            ),
                            handler=callbacks.build_turn_state,
                        ),
                    ),
                    max_iterations=self._max_turns,
                    initial_state={
                        "transcript": [],
                        "last_message_from_a": "(none)",
                        "last_message_from_b": "(none)",
                        "failure_reason": None,
                        "failure_error": None,
                        "should_continue": True,
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
            request_id=f"{request_id}:conversation_pattern",
            dependencies=dependencies,
        )
        return _build_conversation_result(
            workflow_result=workflow_result,
            runtime_state=runtime_state,
            request_id=request_id,
            dependencies=dependencies,
            max_turns=self._max_turns,
            speaker_a_name=self._speaker_a_name,
            speaker_b_name=self._speaker_b_name,
        )


def _build_conversation_result(
    *,
    workflow_result: ExecutionResult,
    runtime_state: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    max_turns: int,
    speaker_a_name: str,
    speaker_b_name: str,
) -> ExecutionResult:
    """Build final conversation result from workflow output.

    Args:
        workflow_result: Completed workflow result.
        runtime_state: Runtime callback state carrying last model response.
        request_id: Resolved request id.
        dependencies: Normalized dependency mapping.
        max_turns: Configured turn limit.
        speaker_a_name: Display speaker name for A.
        speaker_b_name: Display speaker name for B.

    Returns:
        Final conversation execution result.

    Raises:
        RuntimeError: If expected loop outputs are missing.
    """
    loop_step_result = workflow_result.step_results.get("conversation_loop")
    if loop_step_result is None:
        raise RuntimeError("Conversation loop step result is missing.")

    loop_output = normalize_mapping(loop_step_result.output)
    final_state = normalize_mapping(loop_output.get("final_state"))
    transcript = normalize_mapping_records(final_state.get("transcript"))
    failure_reason = _normalize_optional_text(final_state.get("failure_reason"))
    failure_error = _normalize_optional_text(final_state.get("failure_error"))
    workflow_payload = workflow_result.asdict()
    workflow_artifacts = workflow_result.output.get("artifacts", [])
    runtime_last_response = runtime_state.get("last_model_response")
    model_response = (
        runtime_last_response if isinstance(runtime_last_response, LLMResponse) else None
    )

    if failure_reason is not None:
        return build_failure_result(
            error=failure_error or "Conversation loop failed.",
            model_response=model_response,
            tool_results=[],
            request_id=request_id,
            dependencies=dependencies,
            metadata={"mode": "conversation_pattern", "stage": "loop"},
            output={
                "terminated_reason": failure_reason,
                "participants": {
                    "speaker_a": speaker_a_name,
                    "speaker_b": speaker_b_name,
                },
                "transcript": transcript,
                "turns_executed": _safe_int(loop_output.get("iterations_executed")),
                "final_output": _build_final_output(transcript),
                "workflow": workflow_payload,
                "artifacts": workflow_artifacts,
            },
        )

    if not workflow_result.success:
        return build_failure_result(
            error="Conversation workflow failed.",
            model_response=model_response,
            tool_results=[],
            request_id=request_id,
            dependencies=dependencies,
            metadata={"mode": "conversation_pattern", "stage": "workflow"},
            output={
                "terminated_reason": "workflow_failure",
                "participants": {
                    "speaker_a": speaker_a_name,
                    "speaker_b": speaker_b_name,
                },
                "transcript": transcript,
                "turns_executed": _safe_int(loop_output.get("iterations_executed")),
                "final_output": _build_final_output(transcript),
                "workflow": workflow_payload,
                "artifacts": workflow_artifacts,
            },
        )

    turns_executed = _safe_int(loop_output.get("iterations_executed"))
    if turns_executed < 1 and transcript:
        turns_executed = max(1, len(transcript) // 2)

    return ExecutionResult(
        output={
            "participants": {
                "speaker_a": speaker_a_name,
                "speaker_b": speaker_b_name,
            },
            "transcript": transcript,
            "turns_executed": turns_executed,
            "final_output": _build_final_output(transcript),
            "terminated_reason": "completed",
            "workflow": workflow_payload,
            "artifacts": workflow_artifacts,
        },
        success=True,
        tool_results=[],
        model_response=model_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            "mode": "conversation_pattern",
            "max_turns": max_turns,
            "turns_executed": turns_executed,
        },
    )


def _resolve_turn_context(
    context: Mapping[str, object],
) -> tuple[int, list[dict[str, object]], str, str]:
    """Resolve turn metadata and normalized loop state for one conversation iteration."""
    loop_meta = context.get("_loop")
    turn_number = 1
    if isinstance(loop_meta, Mapping):
        turn_number = max(1, _safe_int(loop_meta.get("iteration", 1)))

    loop_state = context.get("loop_state")
    state_mapping = loop_state if isinstance(loop_state, Mapping) else {}
    transcript = normalize_mapping_records(state_mapping.get("transcript"))
    last_message_from_a = str(state_mapping.get("last_message_from_a", "(none)"))
    last_message_from_b = str(state_mapping.get("last_message_from_b", "(none)"))
    return turn_number, transcript, last_message_from_a, last_message_from_b


def _extract_delegate_batch_call_result(
    *,
    context: Mapping[str, object],
    dependency_step_id: str,
    call_id: str,
) -> Mapping[str, object] | None:
    """Extract one call-result mapping from a dependency ``DelegateBatchStep`` output."""
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return None
    dependency_payload = dependency_results.get(dependency_step_id)
    if not isinstance(dependency_payload, Mapping):
        return None
    dependency_output = dependency_payload.get("output")
    if not isinstance(dependency_output, Mapping):
        return None
    results = dependency_output.get("results")
    if not isinstance(results, list):
        return None
    for result_entry in results:
        if not isinstance(result_entry, Mapping):
            continue
        if str(result_entry.get("call_id", "")) != call_id:
            continue
        return result_entry
    return None


def _extract_call_output(call_result: Mapping[str, object] | None) -> dict[str, object]:
    """Extract normalized delegate output mapping from one call-result entry."""
    if not isinstance(call_result, Mapping):
        return {}
    output = call_result.get("output")
    if isinstance(output, Mapping):
        return dict(output)
    return {}


def _extract_model_text_from_output(output: Mapping[str, object]) -> str:
    """Extract one normalized text payload from delegate output."""
    model_text = output.get("model_text")
    if isinstance(model_text, str):
        return model_text.strip()
    final_output = output.get("final_output")
    if isinstance(final_output, str):
        return final_output.strip()
    if isinstance(final_output, Mapping):
        text_value = final_output.get("message")
        if isinstance(text_value, str):
            return text_value.strip()
    return ""


def _extract_call_model_response(call_result: Mapping[str, object] | None) -> LLMResponse | None:
    """Deserialize model response payload from one call-result entry when present."""
    if not isinstance(call_result, Mapping):
        return None
    model_response = call_result.get("model_response")
    if isinstance(model_response, LLMResponse):
        return model_response
    if isinstance(model_response, Mapping):
        try:
            return LLMResponse(**dict(model_response))
        except TypeError:
            return None
    return None


def _is_call_success(call_result: Mapping[str, object] | None) -> bool:
    """Return whether one call-result entry succeeded."""
    if not isinstance(call_result, Mapping):
        return False
    return bool(call_result.get("success", False))


def _extract_call_error(
    call_result: Mapping[str, object] | None,
    *,
    fallback_message: str,
) -> str:
    """Extract one human-readable error message from a call-result entry."""
    if not isinstance(call_result, Mapping):
        return fallback_message
    raw_error = call_result.get("error")
    if isinstance(raw_error, str) and raw_error.strip():
        return raw_error.strip()
    output = call_result.get("output")
    if isinstance(output, Mapping):
        output_error = output.get("error")
        if isinstance(output_error, str) and output_error.strip():
            return output_error.strip()
    return fallback_message


def _render_conversation_prompt(
    *,
    template_text: str,
    field_name: str,
    task_prompt: str,
    turn_number: int,
    speaker_name: str,
    partner_name: str,
    partner_message: str,
    transcript: list[dict[str, object]],
) -> str:
    """Render one speaker prompt from template and conversation state.

    Args:
        template_text: Prompt template text.
        field_name: Field label used for validation and error messages.
        task_prompt: Shared task prompt for the run.
        turn_number: One-based turn number.
        speaker_name: Active speaker display name.
        partner_name: Partner speaker display name.
        partner_message: Most recent partner message text.
        transcript: Conversation transcript entries.

    Returns:
        Rendered speaker prompt text.
    """
    return render_prompt_template(
        template_text=template_text,
        variables={
            "task_prompt": task_prompt,
            "turn": turn_number,
            "speaker_name": speaker_name,
            "partner_name": partner_name,
            "partner_message": partner_message,
            "conversation_transcript_json": json.dumps(
                transcript,
                ensure_ascii=True,
                sort_keys=True,
            ),
        },
        field_name=field_name,
    )


def _build_loop_failure_state(
    *,
    transcript: list[dict[str, object]],
    last_message_from_a: str,
    last_message_from_b: str,
    failure_reason: str,
    failure_error: str,
) -> dict[str, object]:
    """Build loop state snapshot for one failed speaker invocation.

    Args:
        transcript: Transcript entries accumulated so far.
        last_message_from_a: Last speaker A message.
        last_message_from_b: Last speaker B message.
        failure_reason: Failure reason identifier.
        failure_error: Human-readable failure error.

    Returns:
        Updated loop state payload.
    """
    return {
        "transcript": transcript,
        "last_message_from_a": last_message_from_a,
        "last_message_from_b": last_message_from_b,
        "failure_reason": failure_reason,
        "failure_error": failure_error,
        "should_continue": False,
    }


def _extract_model_text(result: ExecutionResult) -> str:
    """Extract one normalized model-text field from an agent result.

    Args:
        result: Agent execution result.

    Returns:
        Trimmed model text fallback.
    """
    model_text = result.output.get("model_text")
    if isinstance(model_text, str):
        return model_text.strip()
    final_output = result.output.get("final_output")
    if isinstance(final_output, str):
        return final_output.strip()
    return ""


def _extract_failure_error(
    result: ExecutionResult,
    *,
    fallback_message: str,
) -> str:
    """Extract one human-readable failure message from agent result output.

    Args:
        result: Agent execution result.
        fallback_message: Fallback message when no explicit error is available.

    Returns:
        Failure error text.
    """
    raw_error = result.output.get("error")
    if isinstance(raw_error, str) and raw_error.strip():
        return raw_error.strip()
    return fallback_message


def _build_final_output(transcript: list[dict[str, object]]) -> dict[str, object]:
    """Extract one final output payload from transcript entries.

    Args:
        transcript: Normalized transcript entries.

    Returns:
        Final output payload with last speaker/message, or empty mapping.
    """
    if not transcript:
        return {}
    last_entry = transcript[-1]
    return {
        "speaker": str(last_entry.get("speaker", "")),
        "message": str(last_entry.get("message", "")),
    }


def _normalize_speaker_name(value: str, *, field_name: str) -> str:
    """Validate and normalize one speaker name.

    Args:
        value: Raw speaker name value.
        field_name: Field label used in validation error messages.

    Returns:
        Normalized non-empty speaker name.

    Raises:
        ValueError: Raised when ``value`` is blank.
    """
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _normalize_optional_text(value: object) -> str | None:
    """Normalize optional text value.

    Args:
        value: Raw value.

    Returns:
        Normalized string or ``None``.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _safe_int(value: object) -> int:
    """Convert arbitrary values to integer with deterministic fallback to zero.

    Args:
        value: Raw value.

    Returns:
        Parsed integer value.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


__all__ = ["ConversationPattern"]
