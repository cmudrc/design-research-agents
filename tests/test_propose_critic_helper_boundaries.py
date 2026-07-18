from __future__ import annotations

import pytest

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._implementations._shared._workflow_internal import _propose_critic_helpers as helpers
from design_research_agents._runtime._patterns import WorkflowBudgetTracker


def _callbacks() -> helpers.ProposeCriticLoopCallbacks:
    return helpers.ProposeCriticLoopCallbacks(
        resolved_model="model",
        task_prompt="Improve the design.",
        request_id="request",
        dependencies={},
        proposer_user_prompt_template=helpers.DEFAULT_PROPOSER_USER_PROMPT_TEMPLATE,
        critic_system_prompt=helpers.DEFAULT_CRITIC_SYSTEM_PROMPT,
        critic_user_prompt_template=helpers.DEFAULT_CRITIC_USER_PROMPT_TEMPLATE,
        budget_tracker=WorkflowBudgetTracker(),
    )


def test_propose_critic_iteration_builders_report_model_and_delegate_failures() -> None:
    callbacks = _callbacks()
    model_context = {
        "dependency_results": {
            "propose_critic_proposer": {"output": {"output": {"model_text": "proposal"}}},
            "propose_critic_critic_model": {"success": False, "error": "model failed"},
        }
    }
    delegate_context = {
        "dependency_results": {
            "propose_critic_proposer": {"output": {"output": {"model_text": "proposal"}}},
            "propose_critic_critic_delegate": {"success": False, "error": "delegate failed"},
        }
    }

    assert callbacks.build_iteration_from_model(model_context) == {
        "failure_reason": "iteration_failed",
        "failure_error": "model failed",
        "proposal": "proposal",
    }
    assert callbacks.build_iteration_from_delegate(delegate_context) == {
        "failure_reason": "iteration_failed",
        "failure_error": "delegate failed",
        "proposal": "proposal",
    }


def test_propose_critic_state_reducer_covers_missing_failed_and_valid_iterations() -> None:
    callbacks = _callbacks()
    missing = callbacks.state_reducer({}, ExecutionResult(success=False), 1)
    assert missing["failure_error"] == "Iteration result missing propose_critic_iteration output."

    failed_result = ExecutionResult(
        success=False,
        step_results={
            "propose_critic_iteration": WorkflowStepResult(
                step_id="propose_critic_iteration",
                status="failed",
                success=False,
                error="workflow failed",
            )
        },
    )
    assert callbacks.state_reducer({}, failed_result, 1)["failure_error"] == "workflow failed"

    explicit_failure = ExecutionResult(
        success=False,
        step_results={
            "propose_critic_iteration": WorkflowStepResult(
                step_id="propose_critic_iteration",
                status="completed",
                success=True,
                output={"failure_reason": "critic_failed", "proposal": "draft"},
            )
        },
    )
    reduced_failure = callbacks.state_reducer({"proposal": "old"}, explicit_failure, 2)
    assert reduced_failure["failure_reason"] == "critic_failed"
    assert reduced_failure["failure_error"] == "Critic iteration failed."
    assert reduced_failure["proposal"] == "draft"

    valid_result = ExecutionResult(
        success=True,
        step_results={
            "propose_critic_iteration": WorkflowStepResult(
                step_id="propose_critic_iteration",
                status="completed",
                success=True,
                output={
                    "proposal": "revised",
                    "approved": True,
                    "feedback": "good",
                    "revision_goals": "invalid",
                },
            )
        },
    )
    reduced = callbacks.state_reducer({"critique_iterations": "invalid"}, valid_result, 3)
    assert reduced["approved"] is True
    assert reduced["revision_goals"] == []
    assert reduced["critique_iterations"] == [
        {
            "iteration": 3,
            "proposal": "revised",
            "approved": True,
            "feedback": "good",
            "revision_goals": [],
        }
    ]


def test_propose_critic_callback_validation_and_response_recording_edges() -> None:
    callbacks = _callbacks()
    with pytest.raises(ValueError, match="Loop metadata is required"):
        callbacks._extract_iteration_state({})

    callbacks._record_step_model_response({"output": {"model_response": {"unknown": True}}})
    assert callbacks.last_model_response is None
    assert callbacks._build_critique_result(proposal="draft", parsed_critique=None)["failure_reason"] == (
        "critic_invalid_json"
    )
    invalid_schema = callbacks._build_critique_result(proposal="draft", parsed_critique={"approved": True})
    assert invalid_schema["failure_reason"] == "critic_invalid_schema"


def test_propose_critic_payload_extractors_normalize_malformed_shapes() -> None:
    assert helpers._extract_dependency_output(context={}, dependency_id="step") == {}
    assert (
        helpers._extract_dependency_output(
            context={"dependency_results": {"step": "invalid"}},
            dependency_id="step",
        )
        == {}
    )
    assert helpers._extract_step_delegate_output({}) == {}
    assert helpers._extract_step_delegate_output({"output": {"output": "invalid"}}) == {}
    assert helpers._extract_step_model_response({}) is None
    assert helpers._extract_step_model_response({"output": {"model_response": "invalid"}}) is None
    assert helpers._extract_step_model_response({"output": {"model_response": {"unknown": True}}}) is None

    assert helpers._extract_delegate_text({"final_output": "text"}) == "text"
    assert helpers._extract_delegate_text({"final_output": {"b": 2, "a": 1}}) == '{"a": 1, "b": 2}'
    assert helpers._extract_delegate_text({"model_text": "model"}) == "model"
    assert helpers._extract_delegate_text({"value": 3}) == '{"value": 3}'
