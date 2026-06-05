from __future__ import annotations

from types import SimpleNamespace

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult
from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._implementations._patterns import _debate_pattern as debate_impl
from design_research_agents._implementations._patterns import _plan_execute_pattern as plan_impl
from design_research_agents._implementations._patterns import _ralph_loop_pattern as ralph_impl
from design_research_agents._implementations._patterns import _round_based_coordination_pattern as round_impl
from design_research_agents._implementations._patterns import _router_delegate_pattern as router_impl
from design_research_agents._implementations._patterns import _two_speaker_conversation_pattern as conv_impl
from design_research_agents._implementations._shared._workflow_internal import (
    _propose_critic_helpers as critic_helpers,
)
from design_research_agents._runtime._patterns import WorkflowBudgetTracker
from design_research_agents._skills import SkillsConfig
from design_research_agents._skills._models import SkillCatalog, SkillsContext
from design_research_agents.tools import Toolbox
from tests.helpers.workflow_stubs import SequenceLLMClient


def test_ralph_private_output_helpers_cover_compaction_and_numeric_edges() -> None:
    answer = object()
    roles = (
        ralph_impl.RalphLoopPattern.RoleSpec(role_id="writer", delegate=object()),
        ralph_impl.RalphLoopPattern.RoleSpec(role_id="evaluator", delegate=object()),
    )
    role_results = {
        "writer": {"output": {"final_output": {"answer": answer}}},
        "evaluator": {"output": {"score": 0.9}},
        "bad": object(),
    }

    assert ralph_impl._resolve_synthesized_output(
        role_results,
        ordered_roles=roles,
        evaluator_role_id="evaluator",
    ) == {"final_output": {"answer": answer}}
    assert ralph_impl._resolve_synthesized_output(
        {"evaluator": {"output": {"score": 0.8}}},
        ordered_roles=roles,
        evaluator_role_id="evaluator",
    ) == {"score": 0.8}
    assert ralph_impl._resolve_synthesized_output({}, ordered_roles=roles, evaluator_role_id="evaluator") == {}

    compacted = ralph_impl._compact_role_results(
        {
            "writer": {
                "success": True,
                "output": {
                    "workflow": {"large": True},
                    "artifacts": [],
                    "model_text": '{"text": "json"}',
                    "notes": ["a", object()],
                },
            },
            "reviewer": {"output": {"model_text": " plain text "}},
            "scorer": {"output": {"score": 2.0}},
            "skip": object(),
        }
    )
    assert compacted["writer"]["output"] == {"text": "json"}
    assert compacted["reviewer"]["output"] == {"text": "plain text"}
    assert ralph_impl._compact_role_output({"notes": ["a", object()], "count": 2})["notes"][1].startswith("<object")
    assert ralph_impl._mapping_from_compact_value("[]") == {"text": "[]"}
    assert ralph_impl._mapping_from_compact_value(" ") == {}
    assert ralph_impl._mapping_from_compact_value(True) == {"value": True}
    assert ralph_impl._mapping_from_compact_value(object()) == {}
    assert ralph_impl._as_list(("not", "a", "list")) == []

    assert ralph_impl._extract_score({"score": 2.0}) == 1.0
    assert ralph_impl._extract_score({"score": -1.0}) == 0.0
    assert ralph_impl._extract_score({"model_text": '{"score": 0.4}'}) == 0.4
    assert ralph_impl._extract_score({"model_text": '{"score": "bad"}'}) is None
    assert ralph_impl._extract_score({"model_text": "not-json"}) is None
    assert ralph_impl._safe_float(True) == 1.0
    assert ralph_impl._safe_float("bad") == 0.0
    assert ralph_impl._safe_int(True) == 1
    assert ralph_impl._safe_int(2.8) == 2
    assert ralph_impl._safe_int("bad") == 0

    with pytest.raises(ValueError, match="at least one"):
        ralph_impl.RalphLoopPattern(roles=(), evaluator_role_id="evaluator")
    with pytest.raises(ValueError, match="non-empty"):
        ralph_impl.RalphLoopPattern(
            roles=(ralph_impl.RalphLoopPattern.RoleSpec(role_id=" ", delegate=object()),),
            evaluator_role_id="evaluator",
        )
    with pytest.raises(ValueError, match="unique"):
        ralph_impl.RalphLoopPattern(
            roles=(
                ralph_impl.RalphLoopPattern.RoleSpec(role_id="dup", delegate=object()),
                ralph_impl.RalphLoopPattern.RoleSpec(role_id="dup", delegate=object()),
            ),
            evaluator_role_id="dup",
        )


def test_plan_execute_private_helpers_cover_payload_parsing_and_validation() -> None:
    llm_client = SequenceLLMClient(response_texts=['{"steps": []}'])
    with pytest.raises(ValueError, match="max_iterations"):
        plan_impl.PlanExecutePattern(llm_client=llm_client, tool_runtime=Toolbox(), max_iterations=0)
    with pytest.raises(ValueError, match="max_tool_calls"):
        plan_impl.PlanExecutePattern(llm_client=llm_client, tool_runtime=Toolbox(), max_tool_calls_per_step=0)

    assert plan_impl._extract_planner_payload({"steps": []}) == {"steps": []}
    assert plan_impl._extract_planner_payload({"final_output": {"steps": [{"step_id": "s"}]}}) == {
        "steps": [{"step_id": "s"}]
    }
    assert plan_impl._extract_planner_payload({"final_output": '{"steps": []}'}) == {"steps": []}
    assert plan_impl._extract_planner_payload({"model_text": '{"steps": []}'}) == {"steps": []}
    assert plan_impl._extract_planner_payload({"final_output": "not-json"}) is None
    assert plan_impl._extract_planner_payload({}) is None

    parsed = plan_impl._parse_planner_model_response(LLMResponse(text='{"steps": []}'), {})
    assert parsed == {"plan": {"steps": []}}
    assert plan_impl._extract_model_response_from_model_step_output({"model_response": {"text": "ok"}}) == LLMResponse(
        text="ok"
    )
    assert plan_impl._extract_model_response_from_model_step_output({"model_response": {"provider": "missing"}}) is None
    assert plan_impl._deserialize_model_response({"text": "ok"}) == LLMResponse(text="ok")
    assert plan_impl._deserialize_model_response("bad") is None
    assert plan_impl._deserialize_model_response({"provider": "missing"}) is None


def _workflow_result_for_conversation(
    *,
    workflow_success: bool,
    loop_output: dict[str, object],
) -> ExecutionResult:
    loop_step = SimpleNamespace(output=loop_output, success=workflow_success, error=None, metadata={})
    return ExecutionResult(
        success=workflow_success,
        output={"artifacts": [{"path": "trace.json"}]},
        step_results={"conversation_loop": loop_step},
    )


def test_two_speaker_conversation_helpers_cover_result_branches() -> None:
    with pytest.raises(RuntimeError, match="Conversation loop"):
        conv_impl._build_conversation_result(
            workflow_result=ExecutionResult(success=True),
            runtime_state={},
            request_id="req",
            dependencies={},
            max_turns=2,
            speaker_a_name="A",
            speaker_b_name="B",
            skills_context=None,
        )

    failure_result = conv_impl._build_conversation_result(
        workflow_result=_workflow_result_for_conversation(
            workflow_success=True,
            loop_output={
                "iterations_executed": 0,
                "final_state": {
                    "transcript": [{"speaker": "A", "message": "Hi"}, {"speaker": "B", "message": "No"}],
                    "failure_reason": "speaker_failed",
                    "failure_error": "bad turn",
                },
            },
        ),
        runtime_state={"last_model_response": LLMResponse(text="last")},
        request_id="req",
        dependencies={"dep": True},
        max_turns=2,
        speaker_a_name="A",
        speaker_b_name="B",
        skills_context=None,
    )
    assert failure_result.success is False
    assert failure_result.output["terminated_reason"] == "speaker_failed"
    assert failure_result.model_response == LLMResponse(text="last")

    workflow_failure = conv_impl._build_conversation_result(
        workflow_result=_workflow_result_for_conversation(
            workflow_success=False,
            loop_output={"iterations_executed": 1, "final_state": {"transcript": []}},
        ),
        runtime_state={},
        request_id="req",
        dependencies={},
        max_turns=2,
        speaker_a_name="A",
        speaker_b_name="B",
        skills_context=None,
    )
    assert workflow_failure.output["terminated_reason"] == "workflow_failure"

    success = conv_impl._build_conversation_result(
        workflow_result=_workflow_result_for_conversation(
            workflow_success=True,
            loop_output={
                "iterations_executed": 1,
                "final_state": {"transcript": [{"speaker": "A", "message": "Done"}]},
            },
        ),
        runtime_state={},
        request_id="req",
        dependencies={},
        max_turns=2,
        speaker_a_name="A",
        speaker_b_name="B",
        skills_context=None,
    )
    assert success.success is True
    assert success.output["final_output"] == {"speaker": "A", "message": "Done"}

    turn, transcript, last_a, last_b = conv_impl._resolve_turn_context(
        {
            "_loop": {"iteration": "3"},
            "loop_state": {
                "transcript": [{"speaker": "A", "message": "Hi"}],
                "last_message_from_a": "Hi",
                "last_message_from_b": "Hello",
            },
        }
    )
    assert (turn, transcript, last_a, last_b) == (3, [{"speaker": "A", "message": "Hi"}], "Hi", "Hello")
    assert conv_impl._extract_model_text_from_output({"model_text": " text "}) == "text"
    assert conv_impl._extract_model_text_from_output({"final_output": {"message": " nested "}}) == "nested"
    assert conv_impl._extract_model_text_from_output({}) == ""
    assert (
        conv_impl._render_conversation_prompt(
            template_text=(
                "$task_prompt $turn $speaker_name $partner_name $partner_message $conversation_transcript_json"
            ),
            field_name="speaker_prompt",
            task_prompt="Task",
            turn_number=1,
            speaker_name="A",
            partner_name="B",
            partner_message="Hi",
            transcript=[],
        )
        == "Task 1 A B Hi []"
    )
    failure_state = conv_impl._build_loop_failure_state(
        transcript=[],
        last_message_from_a="a",
        last_message_from_b="b",
        failure_reason="failed",
        failure_error="bad",
    )
    assert failure_state["should_continue"] is False
    assert conv_impl._extract_model_text(ExecutionResult(success=True, output={"final_output": " final "})) == "final"
    assert (
        conv_impl._extract_failure_error(
            ExecutionResult(success=False, output={"error": " err "}),
            fallback_message="fallback",
        )
        == "err"
    )
    assert conv_impl._extract_failure_error(ExecutionResult(success=False), fallback_message="fallback") == "fallback"
    assert conv_impl._build_final_output([]) == {}
    with pytest.raises(ValueError, match="speaker"):
        conv_impl._normalize_speaker_name(" ", field_name="speaker")
    assert conv_impl._normalize_optional_text(3) is None
    assert conv_impl._normalize_optional_text(" ") is None
    assert conv_impl._safe_int(True) == 1
    assert conv_impl._safe_int(2.9) == 2
    assert conv_impl._safe_int("bad") == 0


def _propose_callbacks() -> critic_helpers.ProposeCriticLoopCallbacks:
    return critic_helpers.ProposeCriticLoopCallbacks(
        resolved_model="model",
        task_prompt="Solve it",
        request_id="req",
        dependencies={},
        proposer_user_prompt_template="Task $task_prompt iteration $iteration: $prior_feedback $revision_goals_json",
        critic_system_prompt="Critic",
        critic_user_prompt_template="Task $task_prompt proposal $proposal",
        budget_tracker=WorkflowBudgetTracker(),
    )


def _proposer_step(text: str) -> dict[str, object]:
    return {
        "success": True,
        "output": {
            "output": {"final_output": text},
            "model_response": {"text": text, "latency_ms": 5},
        },
    }


def test_propose_critic_callbacks_cover_failure_and_success_paths() -> None:
    callbacks = _propose_callbacks()
    assert callbacks.continue_predicate(1, {"approved": True}) is False
    assert callbacks.continue_predicate(1, {"failure_reason": "bad"}) is False
    assert callbacks.continue_predicate(1, {}) is True

    loop_context = {"_loop": {"iteration": "2"}, "loop_state": {"feedback": "fix", "revision_goals": ["goal"]}}
    assert "iteration 2" in callbacks.build_proposer_prompt(loop_context)
    with pytest.raises(ValueError, match="Loop metadata"):
        callbacks.build_proposer_prompt({})

    context_with_proposer = {"dependency_results": {"propose_critic_proposer": _proposer_step("proposal")}}
    request = callbacks.build_critic_request(context_with_proposer)
    assert request.messages[-1].content.endswith("proposal proposal")
    assert callbacks.parse_critic_model_response(LLMResponse(text='{"approved": true}'), {}) == {
        "critique": {"approved": True}
    }

    model_failure = callbacks.build_iteration_from_model(
        {
            "dependency_results": {
                "propose_critic_proposer": _proposer_step("proposal"),
                "propose_critic_critic_model": {"success": False, "error": "model failed"},
            }
        }
    )
    assert model_failure["failure_error"] == "model failed"

    model_success = callbacks.build_iteration_from_model(
        {
            "dependency_results": {
                "propose_critic_proposer": _proposer_step("proposal"),
                "propose_critic_critic_model": {
                    "success": True,
                    "output": {
                        "parsed": {
                            "critique": {
                                "approved": False,
                                "feedback": "revise",
                                "revision_goals": ["clarify"],
                            }
                        },
                        "model_response": {"text": "critique", "latency_ms": 7},
                    },
                },
            }
        }
    )
    assert model_success["feedback"] == "revise"
    assert callbacks.last_model_response == LLMResponse(text="critique", latency_ms=7)

    delegate_failure = callbacks.build_iteration_from_delegate(
        {
            "dependency_results": {
                "propose_critic_proposer": _proposer_step("proposal"),
                "propose_critic_critic_delegate": {"success": False, "error": "delegate failed"},
            }
        }
    )
    assert delegate_failure["failure_error"] == "delegate failed"

    delegate_success = callbacks.build_iteration_from_delegate(
        {
            "dependency_results": {
                "propose_critic_proposer": _proposer_step("proposal"),
                "propose_critic_critic_delegate": {
                    "success": True,
                    "output": {
                        "output": {"final_output": '{"approved": true, "feedback": "ok", "revision_goals": []}'}
                    },
                },
            }
        }
    )
    assert delegate_success["approved"] is True

    assert (
        callbacks._build_critique_result(proposal="p", parsed_critique=None)["failure_reason"] == "critic_invalid_json"
    )
    assert (
        callbacks._build_critique_result(proposal="p", parsed_critique={"approved": "yes"})["failure_reason"]
        == "critic_invalid_schema"
    )
    assert callbacks._extract_iteration_state(loop_context) == (2, "fix", ["goal"])

    missing_step = callbacks.state_reducer({}, ExecutionResult(success=True), 1)
    assert missing_step["failure_error"].startswith("Iteration result missing")
    failed_step = callbacks.state_reducer(
        {},
        ExecutionResult(
            success=False,
            step_results={"propose_critic_iteration": SimpleNamespace(success=False, error="boom", output={})},
        ),
        1,
    )
    assert failed_step["failure_error"] == "boom"
    iteration_failure = callbacks.state_reducer(
        {},
        ExecutionResult(
            success=True,
            step_results={
                "propose_critic_iteration": SimpleNamespace(
                    success=True,
                    error=None,
                    output={"failure_reason": "critic_invalid_json", "proposal": "p"},
                )
            },
        ),
        1,
    )
    assert iteration_failure["failure_reason"] == "critic_invalid_json"
    reduced = callbacks.state_reducer(
        {},
        ExecutionResult(
            success=True,
            step_results={
                "propose_critic_iteration": SimpleNamespace(
                    success=True,
                    error=None,
                    output={
                        "proposal": "p",
                        "approved": True,
                        "feedback": "ok",
                        "revision_goals": ["done"],
                    },
                )
            },
        ),
        1,
    )
    assert reduced["critique_iterations"][0]["approved"] is True
    assert critic_helpers._extract_delegate_text({"final_output": {"answer": 1}}) == '{"answer": 1}'
    assert critic_helpers._extract_delegate_text({"model_text": "text"}) == "text"
    assert critic_helpers._extract_delegate_text({"other": 1}) == '{"other": 1}'


def test_debate_private_helpers_cover_result_and_extraction_branches() -> None:
    assert debate_impl._extract_debate_round_state(ExecutionResult(success=True)) == {}
    assert debate_impl._resolve_round_context(
        {
            "_loop": {"iteration": "2"},
            "loop_state": {
                "rounds": [{"round": 1}, "skip"],
                "prior_affirmative_argument": "yes",
                "prior_negative_argument": "no",
            },
        }
    ) == (2, [{"round": 1}], "yes", "no")
    assert debate_impl._extract_rounds_from_context({}) == []
    assert debate_impl._extract_rounds_from_context(
        {"dependency_results": {"debate_rounds": {"output": {"final_state": {"rounds": [{"round": 1}]}}}}}
    ) == [{"round": 1}]
    assert debate_impl._extract_dependency_output({}, dependency_id="missing") == {}
    assert debate_impl._extract_dependency_output(
        {"dependency_results": {"dep": {"output": {"value": 1}}}},
        dependency_id="dep",
    ) == {"value": 1}
    response = LLMResponse(text="ok")
    assert debate_impl._extract_model_response_from_model_step_output({"model_response": response}) is response
    assert debate_impl._extract_model_response_from_model_step_output({"model_response": {"text": "ok"}}) == response
    assert debate_impl._extract_model_response_from_model_step_output({"model_response": {"provider": "bad"}}) is None
    assert debate_impl._extract_model_text_from_output({"final_output": {"message": " nested "}}) == "nested"
    assert debate_impl._extract_delegate_verdict({"winner": "affirmative", "rationale": "r", "synthesis": "s"}) == {
        "winner": "affirmative",
        "rationale": "r",
        "synthesis": "s",
    }
    assert debate_impl._extract_delegate_verdict({"final_output": '{"winner": "a"}'}) == {"winner": "a"}
    assert debate_impl._extract_delegate_verdict({"model_text": "not-json"}) is None
    assert debate_impl._safe_int(True) == 1
    assert debate_impl._safe_int(2.8) == 2
    assert debate_impl._safe_int("bad") == 1

    round_failure = debate_impl._build_debate_result(
        workflow_result=ExecutionResult(
            success=True,
            output={"artifacts": [{"path": "round.json"}]},
            step_results={
                "debate_rounds": SimpleNamespace(
                    output={
                        "final_state": {
                            "failure_reason": "speaker_failed",
                            "failure_error": "bad round",
                            "rounds": [{"round": 1}],
                        }
                    }
                )
            },
        ),
        runtime_state={"last_model_response": LLMResponse(text="last")},
        request_id="req",
        dependencies={},
        skills_context=None,
    )
    assert round_failure.success is False
    assert round_failure.output["terminated_reason"] == "speaker_failed"
    assert round_failure.output["error"] == "bad round"

    judge_failure = debate_impl._build_debate_result(
        workflow_result=ExecutionResult(
            success=True,
            output={},
            step_results={
                "debate_rounds": SimpleNamespace(output={"final_state": {"rounds": []}}),
                "debate_judge": SimpleNamespace(output={"status": "delegate_failed"}),
                "debate_judge_delegate": SimpleNamespace(error="delegate down"),
            },
        ),
        runtime_state={},
        request_id="req",
        dependencies={},
        skills_context=None,
    )
    assert judge_failure.success is False
    assert judge_failure.output["terminated_reason"] == "judge_invalid_json"
    assert judge_failure.output["error"] == "delegate down"

    success = debate_impl._build_debate_result(
        workflow_result=ExecutionResult(
            success=True,
            output={},
            step_results={
                "debate_rounds": SimpleNamespace(output={"final_state": {"rounds": [{"round": 1}]}}),
                "debate_judge": SimpleNamespace(
                    output={
                        "status": "completed",
                        "verdict": {"winner": "affirmative", "rationale": "clear", "synthesis": "done"},
                    }
                ),
            },
        ),
        runtime_state={},
        request_id="req",
        dependencies={},
        skills_context=None,
    )
    assert success.success is True
    assert success.output["final_output"]["winner"] == "affirmative"


def test_round_based_coordination_helpers_cover_payload_normalization_edges() -> None:
    assert round_impl._normalize_peer_contribution(
        peer_id="p1",
        peer_output={"message": "hello", "proposal": "build", "decision": True, "stop": True},
        round_number=2,
    ) == {
        "peer_id": "p1",
        "round": 2,
        "messages": ["hello"],
        "proposals": {"proposal": "build"},
        "decisions": {"decision": True},
        "stop": True,
    }
    assert round_impl._normalize_peer_contribution(
        peer_id="p2",
        peer_output={"final_output": '```json\n{"messages": ["a"], "proposals": {"x": 1}}\n```'},
        round_number=1,
    )["proposals"] == {"x": 1}
    assert round_impl._normalize_peer_contribution(
        peer_id="p3",
        peer_output={"model_text": "plain text"},
        round_number=1,
    )["messages"] == ["plain text"]
    assert round_impl._normalize_peer_contribution(peer_id="p4", peer_output={}, round_number=1)["messages"] == ["{}"]
    assert round_impl._coerce_contribution_mapping({"proposal": {"a": object()}, "decision": ["x"]})["decisions"] == {
        "decision": ["x"]
    }
    assert round_impl._strip_json_code_fence("```json") == "```json"
    assert round_impl._normalize_singular_contribution("", fallback_key="proposal") == {}
    assert round_impl._normalize_singular_contribution([object()], fallback_key="proposal")["proposal"][0].startswith(
        "<object"
    )
    assert len(round_impl._compute_state_hash({"messages": ["a"], "proposals": {}, "decisions": {}})) == 64
    assert round_impl._safe_int(True) == 1
    assert round_impl._safe_int(3.8) == 3
    assert round_impl._safe_int("bad") == 0
    json_ready = round_impl._json_ready(({"x": object()},))
    assert isinstance(json_ready, list)
    assert isinstance(json_ready[0], dict)
    assert str(json_ready[0]["x"]).startswith("<object")


def test_router_delegate_private_helpers_and_finalization_edges() -> None:
    assert router_impl._extract_selected_name_from_router_output({"tool_name": " alpha "}) == "alpha"
    assert (
        router_impl._extract_selected_name_from_router_output(
            {"step_outputs": [{"tool_name": "first"}, object(), {"tool_name": " second "}]}
        )
        == "second"
    )
    assert router_impl._extract_selected_name_from_router_output({"step_outputs": "bad"}) == ""
    assert (
        router_impl._selected_tool_name_from_result(ExecutionResult(success=True, output={"tool_name": "alpha"}))
        == "alpha"
    )
    assert (
        router_impl._append_activated_skill_context(
            prompt="Prompt",
            tool_results=[ToolResult(tool_name="other", ok=True)],
        )
        == "Prompt"
    )
    activated_prompt = router_impl._append_activated_skill_context(
        prompt="Prompt",
        tool_results=[
            ToolResult(
                tool_name="skills.activate",
                ok=True,
                result={
                    "name": "analysis",
                    "description": "Use analysis",
                    "instructions": "Think carefully.",
                    "skill_root": "/tmp/skill",
                    "compatibility": ["router"],
                },
            )
        ],
    )
    assert "Activated routing skill" in activated_prompt

    skills_context = SkillsContext(
        config=SkillsConfig(project_root=".", allow_automatic_activation=True, pinned_skills=("analysis",)),
        catalog=SkillCatalog(skills=()),
        pinned_skills=(),
    )
    assert router_impl._pinned_only_skills_context(skills_context).config.allow_automatic_activation is False

    pattern = router_impl.RouterDelegatePattern(
        llm_client=SequenceLLMClient(response_texts=[]),
        tool_runtime=Toolbox(),
        alternatives={"alpha": object()},
    )
    budget_tracker = WorkflowBudgetTracker()
    selection_step = WorkflowStepResult(
        step_id="agent_routing_selection",
        status="completed",
        success=True,
        output={"selected_name": "alpha", "selected_step_id": "agent_routing_delegate_alpha"},
    )
    workflow_result = ExecutionResult(
        success=True,
        output={"artifacts": [{"path": "workflow.json"}]},
        step_results={"agent_routing_selection": selection_step},
        execution_order=["agent_routing_selection"],
    )

    with pytest.raises(RuntimeError, match="workflow graph"):
        pattern._finalize_agent_routing_result(
            workflow_result=ExecutionResult(success=False),
            budget_tracker=budget_tracker,
            execution_state=router_impl._RoutingExecutionState(),
            request_id="req",
            dependencies={},
        )
    with pytest.raises(RuntimeError, match="selection step"):
        pattern._finalize_agent_routing_result(
            workflow_result=workflow_result,
            budget_tracker=budget_tracker,
            execution_state=router_impl._RoutingExecutionState(),
            request_id="req",
            dependencies={},
        )

    router_failure = pattern._finalize_agent_routing_result(
        workflow_result=workflow_result,
        budget_tracker=budget_tracker,
        execution_state=router_impl._RoutingExecutionState(router_result=ExecutionResult(success=False)),
        request_id="req",
        dependencies={},
    )
    assert router_failure.success is False
    assert router_failure.output["terminated_reason"] == "routing_failure"

    unknown = pattern._finalize_agent_routing_result(
        workflow_result=ExecutionResult(
            success=True,
            step_results={
                "agent_routing_selection": WorkflowStepResult(
                    step_id="agent_routing_selection",
                    status="completed",
                    success=True,
                    output={"selected_name": "missing", "selected_step_id": "agent_routing_unknown"},
                ),
                "agent_routing_unknown": WorkflowStepResult(
                    step_id="agent_routing_unknown",
                    status="completed",
                    success=True,
                    output={"status": "unknown_alternative"},
                ),
            },
        ),
        budget_tracker=budget_tracker,
        execution_state=router_impl._RoutingExecutionState(router_result=ExecutionResult(success=True)),
        request_id="req",
        dependencies={},
    )
    assert unknown.success is False
    assert unknown.output["terminated_reason"] == "unknown_alternative"

    missing_delegate = pattern._finalize_agent_routing_result(
        workflow_result=workflow_result,
        budget_tracker=budget_tracker,
        execution_state=router_impl._RoutingExecutionState(router_result=ExecutionResult(success=True)),
        request_id="req",
        dependencies={},
    )
    assert missing_delegate.success is False
    assert missing_delegate.output["terminated_reason"] == "routing_failure"

    delegated = pattern._finalize_agent_routing_result(
        workflow_result=workflow_result,
        budget_tracker=budget_tracker,
        execution_state=router_impl._RoutingExecutionState(
            router_result=ExecutionResult(success=True, metadata={"routing": {"score": 1}}),
            delegated_result=ExecutionResult(
                success=True,
                output={"final_output": {"answer": 42}, "terminated_reason": "completed"},
                metadata={"delegate": "alpha"},
                tool_results=[ToolResult(tool_name="calc", ok=True, result={"value": 42})],
                model_response=LLMResponse(text="done"),
            ),
        ),
        request_id="req",
        dependencies={},
    )
    assert delegated.success is True
    assert delegated.output["final_output"] == {"answer": 42}
    assert delegated.metadata["router_delegate"]["selected_alternative"] == "alpha"
