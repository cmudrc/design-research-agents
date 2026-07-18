from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._implementations._patterns import _debate_pattern as debate_impl
from design_research_agents._implementations._patterns import (
    _plan_execute_pattern as planner_impl,
)
from design_research_agents._implementations._patterns._debate_pattern import DebatePattern
from design_research_agents._implementations._patterns._plan_execute_pattern import (
    PlanExecutePattern,
)


class _NoopLLMClient:
    def default_model(self) -> str:
        return "noop-model"

    def close(self) -> None:
        return None

    def __enter__(self) -> _NoopLLMClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        return LLMResponse(model=model, text="{}", provider="noop")


class _StaticDelegate:
    def __init__(self, *, success: bool, payload: Mapping[str, object]) -> None:
        # Prebuild immutable ExecutionResult so delegates stay deterministic across calls.
        self._result = ExecutionResult(
            success=success,
            output=dict(payload),
            tool_results=[],
            model_response=None,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return self._result


class _SingleToolRuntime(ToolRuntime):
    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="sum",
                description="Add numbers",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            ),
        )

    def invoke(
        self,
        tool_name: str,
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        # Echo-style runtime keeps pattern tests focused on orchestration, not tool behavior.
        return ToolResult(tool_name=tool_name, ok=True, result=dict(input))

    def close(self) -> None:
        return None

    def __enter__(self) -> _SingleToolRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


def test_plan_execute_pattern_delegate_paths_and_payload_extraction() -> None:
    # Cover payload extraction fallbacks in priority order.
    assert planner_impl._extract_planner_payload({"steps": [{"step_id": "x"}]}) == {"steps": [{"step_id": "x"}]}
    assert planner_impl._extract_planner_payload({"final_output": {"steps": [{"step_id": "y"}]}}) == {
        "steps": [{"step_id": "y"}]
    }
    assert planner_impl._extract_planner_payload(
        {"final_output": json.dumps({"steps": [{"step_id": "z", "instruction": "i", "success_criteria": "c"}]})}
    ) == {"steps": [{"step_id": "z", "instruction": "i", "success_criteria": "c"}]}
    assert planner_impl._extract_planner_payload(
        {
            "model_text": json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "m",
                            "instruction": "i",
                            "success_criteria": "c",
                        }
                    ]
                }
            )
        }
    ) == {"steps": [{"step_id": "m", "instruction": "i", "success_criteria": "c"}]}
    assert planner_impl._extract_planner_payload({"model_text": "not-json"}) is None

    planner_delegate = _StaticDelegate(
        success=True,
        payload={
            "model_text": json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "compute",
                            "instruction": "Compute once",
                            "success_criteria": "Return answer",
                        }
                    ]
                }
            )
        },
    )
    executor_delegate = _StaticDelegate(success=True, payload={"final_output": {"answer": 7}})
    success_pattern = PlanExecutePattern(
        llm_client=_NoopLLMClient(),
        tool_runtime=_SingleToolRuntime(),
        planner_delegate=planner_delegate,
        executor_delegate=executor_delegate,
        max_iterations=2,
    )
    success_result = success_pattern.run("Solve task")
    assert success_result.success is True
    assert success_result.output["details"]["steps_executed"] == 1
    assert success_result.output["final_output"] == {"answer": 7}

    planner_failure_pattern = PlanExecutePattern(
        llm_client=_NoopLLMClient(),
        tool_runtime=_SingleToolRuntime(),
        planner_delegate=_StaticDelegate(success=False, payload={"error": "bad"}),
    )
    planner_failure = planner_failure_pattern.run("Fail planner")
    assert planner_failure.success is False
    assert planner_failure.output["terminated_reason"] == "planner_invalid_json"

    schema_failure_pattern = PlanExecutePattern(
        llm_client=_NoopLLMClient(),
        tool_runtime=_SingleToolRuntime(),
        planner_delegate=_StaticDelegate(
            success=True,
            payload={"steps": [{"step_id": "missing-fields"}]},
        ),
    )
    schema_failure = schema_failure_pattern.run("Fail schema")
    assert schema_failure.success is False
    assert schema_failure.output["terminated_reason"] == "planner_invalid_schema"

    capped_pattern = PlanExecutePattern(
        llm_client=_NoopLLMClient(),
        tool_runtime=_SingleToolRuntime(),
        planner_delegate=_StaticDelegate(
            success=True,
            payload={
                "steps": [
                    {
                        "step_id": "one",
                        "instruction": "first",
                        "success_criteria": "ok",
                    },
                    {
                        "step_id": "two",
                        "instruction": "second",
                        "success_criteria": "ok",
                    },
                ]
            },
        ),
        executor_delegate=_StaticDelegate(success=True, payload={"final_output": {"done": True}}),
        max_iterations=1,
    )
    capped_result = capped_pattern.run("Cap iterations")
    assert capped_result.success is True
    assert capped_result.output["terminated_reason"] == "completed"
    runtime_metadata = capped_result.metadata.get("runtime")
    assert isinstance(runtime_metadata, Mapping)
    plan_metadata = runtime_metadata.get("plan")
    assert isinstance(plan_metadata, Mapping)
    assert plan_metadata["was_truncated"] is True
    assert plan_metadata["scheduled_step_count"] == 1


def test_plan_execute_validation_and_static_callback_helpers_cover_missing_state() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        PlanExecutePattern(llm_client=_NoopLLMClient(), tool_runtime=_SingleToolRuntime(), max_iterations=0)
    with pytest.raises(ValueError, match="max_tool_calls_per_step"):
        PlanExecutePattern(
            llm_client=_NoopLLMClient(),
            tool_runtime=_SingleToolRuntime(),
            max_tool_calls_per_step=0,
        )

    assert (
        planner_impl.PlanExecutePattern._extract_model_planner_plan(
            normalized_output={},
            planner_success=False,
        )
        is None
    )
    assert (
        planner_impl.PlanExecutePattern._extract_model_planner_plan(
            normalized_output={"parsed": "invalid"},
            planner_success=True,
        )
        is None
    )
    assert (
        planner_impl.PlanExecutePattern._extract_model_planner_plan(
            normalized_output={"parsed": {"plan": "invalid"}},
            planner_success=True,
        )
        is None
    )
    assert (
        planner_impl.PlanExecutePattern._extract_delegate_planner_plan(
            normalized_output={},
            planner_success=False,
        )
        is None
    )
    assert (
        planner_impl.PlanExecutePattern._extract_delegate_planner_plan(
            normalized_output={"output": "invalid"},
            planner_success=True,
        )
        is None
    )

    continue_predicate = planner_impl.PlanExecutePattern._build_plan_continue_predicate({"plan_steps": [{}, {}]})
    assert continue_predicate(2, {})
    assert not continue_predicate(3, {})
    assert not planner_impl.PlanExecutePattern._build_plan_continue_predicate({})(1, {})

    prompt_builder = planner_impl.PlanExecutePattern._build_plan_prompt_builder({})
    with pytest.raises(RuntimeError, match="callbacks are unavailable"):
        prompt_builder({})
    reducer = planner_impl.PlanExecutePattern._build_plan_state_reducer({})
    assert reducer({"kept": True}, ExecutionResult(success=False), 1) == {"kept": True}

    assert planner_impl._extract_planner_payload({"final_output": {"value": 1}}) is None
    assert planner_impl._extract_model_response_from_model_step_output({}) is None
    assert planner_impl._extract_model_response_from_model_step_output({"model_response": {"unknown": True}}) is None
    assert planner_impl._deserialize_model_response("invalid") is None
    assert planner_impl._deserialize_model_response({"unknown": True}) is None


def test_plan_execute_result_builder_reports_incomplete_runtime_state() -> None:
    pattern = PlanExecutePattern(llm_client=_NoopLLMClient(), tool_runtime=_SingleToolRuntime())
    workflow_result = ExecutionResult(success=False)
    tracker = planner_impl.WorkflowBudgetTracker()

    with pytest.raises(ValueError, match="input invalid"):
        pattern._build_plan_execute_result(
            workflow_result=workflow_result,
            request_id="request",
            dependencies={},
            budget_tracker=tracker,
            runtime_state={"fatal_stage": "input_build", "fatal_error": "input invalid"},
        )
    with pytest.raises(RuntimeError, match="planner invalid"):
        pattern._build_plan_execute_result(
            workflow_result=workflow_result,
            request_id="request",
            dependencies={},
            budget_tracker=tracker,
            runtime_state={"fatal_stage": "planner", "fatal_error": "planner invalid"},
        )
    with pytest.raises(RuntimeError, match="runtime state is unavailable"):
        pattern._build_plan_execute_result(
            workflow_result=workflow_result,
            request_id="request",
            dependencies={},
            budget_tracker=tracker,
            runtime_state={},
        )


def test_debate_pattern_delegate_and_helper_branches() -> None:
    # Cover direct, final_output, and model_text verdict payload shapes.
    assert debate_impl._extract_delegate_verdict({"winner": "tie", "rationale": "r", "synthesis": "s"}) == {
        "winner": "tie",
        "rationale": "r",
        "synthesis": "s",
    }
    assert debate_impl._extract_delegate_verdict(
        {
            "final_output": {
                "winner": "affirmative",
                "rationale": "r",
                "synthesis": "s",
            }
        }
    ) == {"winner": "affirmative", "rationale": "r", "synthesis": "s"}
    assert debate_impl._extract_delegate_verdict(
        {
            "final_output": json.dumps(
                {
                    "winner": "negative",
                    "rationale": "r",
                    "synthesis": "s",
                }
            )
        }
    ) == {"winner": "negative", "rationale": "r", "synthesis": "s"}
    assert debate_impl._extract_delegate_verdict(
        {
            "model_text": json.dumps(
                {
                    "winner": "tie",
                    "rationale": "r",
                    "synthesis": "s",
                }
            )
        }
    ) == {"winner": "tie", "rationale": "r", "synthesis": "s"}
    assert debate_impl._extract_delegate_verdict({"model_text": "not-json"}) is None

    assert debate_impl._safe_int(True) == 1
    assert debate_impl._safe_int(2.9) == 2
    assert debate_impl._safe_int("7") == 7
    assert debate_impl._safe_int("bad") == 1

    affirmative = _StaticDelegate(success=True, payload={"model_text": "Affirmative"})
    negative = _StaticDelegate(success=True, payload={"model_text": "Negative"})

    success_debate = DebatePattern(
        llm_client=_NoopLLMClient(),
        tool_runtime=_SingleToolRuntime(),
        affirmative_delegate=affirmative,
        negative_delegate=negative,
        judge_delegate=_StaticDelegate(
            success=True,
            payload={
                "final_output": json.dumps(
                    {
                        "winner": "tie",
                        "rationale": "balanced",
                        "synthesis": "do both",
                    }
                )
            },
        ),
        max_rounds=1,
    )
    success_result = success_debate.run("Debate this")
    assert success_result.success is True
    assert success_result.output["final_output"]["winner"] == "tie"

    invalid_json_debate = DebatePattern(
        llm_client=_NoopLLMClient(),
        tool_runtime=_SingleToolRuntime(),
        affirmative_delegate=affirmative,
        negative_delegate=negative,
        judge_delegate=_StaticDelegate(success=False, payload={"error": "judge failed"}),
        max_rounds=1,
    )
    invalid_json_result = invalid_json_debate.run("Debate this")
    assert invalid_json_result.success is False
    assert invalid_json_result.output["terminated_reason"] == "judge_invalid_json"
    assert "judge failed" in str(invalid_json_result.output["error"])

    invalid_schema_debate = DebatePattern(
        llm_client=_NoopLLMClient(),
        tool_runtime=_SingleToolRuntime(),
        affirmative_delegate=affirmative,
        negative_delegate=negative,
        judge_delegate=_StaticDelegate(
            success=True,
            payload={
                "winner": "unknown",
                "rationale": "r",
                "synthesis": "s",
            },
        ),
        max_rounds=1,
    )
    invalid_schema_result = invalid_schema_debate.run("Debate this")
    assert invalid_schema_result.success is False
    assert invalid_schema_result.output["terminated_reason"] == "judge_invalid_schema"


def test_debate_context_and_payload_helpers_cover_malformed_workflow_shapes() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        DebatePattern(llm_client=_NoopLLMClient(), tool_runtime=_SingleToolRuntime(), max_rounds=0)

    state = debate_impl._resolve_round_context(
        {
            "_loop": {"iteration": "bad"},
            "loop_state": {
                "rounds": [{"round": 1}, "invalid"],
                "prior_affirmative_argument": "yes",
                "prior_negative_argument": "no",
            },
        }
    )
    assert state == (1, [{"round": 1}], "yes", "no")

    assert debate_impl._extract_rounds_from_context({}) == []
    assert debate_impl._extract_rounds_from_context({"dependency_results": {"debate_rounds": "invalid"}}) == []
    assert (
        debate_impl._extract_rounds_from_context({"dependency_results": {"debate_rounds": {"output": "invalid"}}}) == []
    )
    assert (
        debate_impl._extract_rounds_from_context(
            {"dependency_results": {"debate_rounds": {"output": {"final_state": "invalid"}}}}
        )
        == []
    )
    assert (
        debate_impl._extract_rounds_from_context(
            {"dependency_results": {"debate_rounds": {"output": {"final_state": {"rounds": "invalid"}}}}}
        )
        == []
    )
    assert debate_impl._extract_rounds_from_context(
        {"dependency_results": {"debate_rounds": {"output": {"final_state": {"rounds": [{"round": 1}, "invalid"]}}}}}
    ) == [{"round": 1}]

    assert debate_impl._extract_dependency_output({}, dependency_id="step") == {}
    assert (
        debate_impl._extract_dependency_output(
            {"dependency_results": {"step": "invalid"}},
            dependency_id="step",
        )
        == {}
    )
    assert (
        debate_impl._extract_dependency_output(
            {"dependency_results": {"step": {"output": "invalid"}}},
            dependency_id="step",
        )
        == {}
    )
    assert debate_impl._extract_model_response_from_model_step_output({"model_response": LLMResponse(text="ok")}) == (
        LLMResponse(text="ok")
    )
    assert debate_impl._extract_model_response_from_model_step_output({"model_response": {"unknown": True}}) is None
    assert debate_impl._extract_model_response_from_model_step_output({}) is None

    assert debate_impl._extract_model_text_from_output({"model_text": " text "}) == "text"
    assert debate_impl._extract_model_text_from_output({"final_output": " final "}) == "final"
    assert debate_impl._extract_model_text_from_output({"final_output": {"message": " nested "}}) == "nested"
    assert debate_impl._extract_model_text_from_output({"final_output": {"message": 3}}) == ""


def test_debate_state_reducer_and_round_state_extraction_cover_missing_steps() -> None:
    assert debate_impl._DebateWorkflowCallbacks.state_reducer({}, ExecutionResult(success=False), 1) == {}
    failed = ExecutionResult(
        success=False,
        step_results={
            "debate_round": WorkflowStepResult(step_id="debate_round", status="failed", success=False),
        },
    )
    assert debate_impl._DebateWorkflowCallbacks.state_reducer({"kept": True}, failed, 1) == {"kept": True}
    valid = ExecutionResult(
        success=True,
        step_results={
            "debate_round": WorkflowStepResult(
                step_id="debate_round",
                status="completed",
                success=True,
                output={"rounds": []},
            ),
        },
    )
    assert debate_impl._DebateWorkflowCallbacks.state_reducer({}, valid, 1) == {"rounds": []}
    assert debate_impl._extract_debate_round_state(ExecutionResult(success=False)) == {}
    assert (
        debate_impl._extract_debate_round_state(
            ExecutionResult(
                success=True,
                step_results={
                    "debate_rounds": WorkflowStepResult(
                        step_id="debate_rounds",
                        status="completed",
                        success=True,
                        output={"final_state": "invalid"},
                    )
                },
            )
        )
        == {}
    )
