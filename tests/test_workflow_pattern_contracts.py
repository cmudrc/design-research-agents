"""Contract tests for reusable workflow pattern classes."""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable

import pytest

from design_research_agents._contracts import ExecutionResult
from design_research_agents._implementations._patterns._agent_routing_pattern import (
    AgentRoutingPattern,
)
from design_research_agents._implementations._patterns._conversation_pattern import (
    ConversationPattern,
)
from design_research_agents._implementations._patterns._debate_pattern import (
    DebatePattern,
)
from design_research_agents._implementations._patterns._networked_blackboard import (
    BlackboardPattern,
    NetworkedPattern,
)
from design_research_agents._implementations._patterns._plan_execute_pattern import (
    PlanExecutePattern,
)
from design_research_agents._implementations._patterns._rag_pattern import (
    RAGPattern,
)
from design_research_agents._implementations._patterns._reflexion_pattern import (
    ReflexionPattern,
)
from design_research_agents._implementations._patterns._tree_search import (
    TreeSearchPattern,
)
from design_research_agents.tools import Toolbox
from design_research_agents.workflow import Workflow
from tests.helpers.workflow_stubs import SequenceLLMClient, StaticMarkerAgent


@pytest.mark.parametrize(
    ("pattern", "prompt"),
    [
        (
            ConversationPattern(
                llm_client_a=SequenceLLMClient(response_texts=["speaker-a", "speaker-b"]),
                max_turns=1,
            ),
            "Hold a short discussion.",
        ),
        (
            DebatePattern(
                llm_client=SequenceLLMClient(
                    response_texts=[
                        "Affirmative argument",
                        "Negative argument",
                        '{"winner":"tie","rationale":"balanced","synthesis":"both sides have merit"}',
                    ]
                ),
                tool_runtime=Toolbox(),
                max_rounds=1,
            ),
            "Debate local model hosting tradeoffs.",
        ),
        (
            PlanExecutePattern(
                llm_client=SequenceLLMClient(
                    response_texts=[
                        '{"steps":[{"step_id":"one","instruction":"Do one thing","success_criteria":"done"}]}'
                    ]
                ),
                tool_runtime=Toolbox(),
                max_iterations=1,
            ),
            "Execute one deterministic step.",
        ),
        (
            ReflexionPattern(
                llm_client=SequenceLLMClient(
                    response_texts=["draft", '{"approved": true, "feedback": "", "revision_goals": []}']
                ),
                tool_runtime=Toolbox(),
                max_iterations=1,
            ),
            "Refine this sentence.",
        ),
        (
            AgentRoutingPattern(
                llm_client=SequenceLLMClient(
                    response_texts=['{"tool_name":"route_a","tool_input":{},"reason":"best fit"}']
                ),
                tool_runtime=Toolbox(),
                alternatives={"route_a": StaticMarkerAgent(marker="route-a")},
            ),
            "Route this request.",
        ),
        (
            NetworkedPattern(
                peers={"peer_a": StaticMarkerAgent(marker="peer-a")},
                max_rounds=1,
            ),
            "Coordinate one round.",
        ),
        (
            BlackboardPattern(
                peers={"peer_a": StaticMarkerAgent(marker="peer-a")},
                max_rounds=1,
                stability_rounds=1,
            ),
            "Coordinate one blackboard round.",
        ),
        (
            TreeSearchPattern(
                generator_delegate=lambda _context: [{"candidate": "x"}],
                evaluator_delegate=lambda _context: 1.0,
                max_depth=1,
                branch_factor=1,
                beam_width=1,
            ),
            "Search one depth.",
        ),
        (
            RAGPattern(
                reasoning_delegate=StaticMarkerAgent(marker="rag"),
                memory_store=None,
                write_back=False,
            ),
            "Reason with retrieval context.",
        ),
    ],
)
def test_exported_patterns_build_workflow_contract(
    pattern: object,
    prompt: str,
) -> None:
    workflow = pattern.build_workflow(  # type: ignore[attr-defined]
        prompt,
        request_id="req-pattern-build",
        dependencies={},
    )
    assert isinstance(workflow, Workflow)
    assert pattern.workflow is workflow  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("pattern_factory", "prompt", "expected_output_keys"),
    [
        (
            lambda: ConversationPattern(
                llm_client_a=SequenceLLMClient(response_texts=["speaker-a", "speaker-b"]),
                max_turns=1,
            ),
            "Hold a short discussion.",
            {"final_output", "terminated_reason", "workflow"},
        ),
        (
            lambda: DebatePattern(
                llm_client=SequenceLLMClient(
                    response_texts=[
                        "Affirmative argument",
                        "Negative argument",
                        '{"winner":"tie","rationale":"balanced","synthesis":"both sides have merit"}',
                    ]
                ),
                tool_runtime=Toolbox(),
                max_rounds=1,
            ),
            "Debate local model hosting tradeoffs.",
            {"final_output", "terminated_reason", "workflow"},
        ),
        (
            lambda: PlanExecutePattern(
                llm_client=SequenceLLMClient(
                    response_texts=[
                        json.dumps(
                            {
                                "steps": [
                                    {
                                        "step_id": "compute",
                                        "instruction": "Count words in 'design research agents'.",
                                        "success_criteria": "Return word count.",
                                    }
                                ]
                            }
                        ),
                        '{"continue": true, "thought": "execute first step"}',
                        "\n".join(
                            [
                                'stats = call_tool("text.word_count", {"text": "design research agents"})',
                                'final_output = {"result": stats["word_count"]}',
                            ]
                        ),
                    ]
                ),
                tool_runtime=Toolbox(),
                max_iterations=2,
            ),
            "Count words in design research agents.",
            {"final_output", "terminated_reason", "workflow"},
        ),
        (
            lambda: ReflexionPattern(
                llm_client=SequenceLLMClient(
                    response_texts=[
                        "Draft v1",
                        json.dumps(
                            {
                                "approved": True,
                                "feedback": "Looks good.",
                                "revision_goals": [],
                            }
                        ),
                    ]
                ),
                tool_runtime=Toolbox(),
                max_iterations=2,
            ),
            "Write a short design summary.",
            {"proposal", "terminated_reason", "workflow"},
        ),
        (
            lambda: AgentRoutingPattern(
                llm_client=SequenceLLMClient(
                    response_texts=['{"tool_name":"alt_two","tool_input":{},"reason":"best fit"}']
                ),
                tool_runtime=Toolbox(),
                alternatives={
                    "alt_one": StaticMarkerAgent(marker="one"),
                    "alt_two": StaticMarkerAgent(marker="two"),
                },
            ),
            "Route this request.",
            {"delegated_output", "final_output", "workflow"},
        ),
        (
            lambda: NetworkedPattern(
                peers={"peer_a": StaticMarkerAgent(marker="peer-a")},
                max_rounds=1,
            ),
            "Coordinate one round.",
            {"blackboard", "final_output", "workflow"},
        ),
        (
            lambda: BlackboardPattern(
                peers={"peer_a": StaticMarkerAgent(marker="peer-a")},
                max_rounds=1,
                stability_rounds=1,
            ),
            "Coordinate one blackboard round.",
            {"blackboard", "final_output", "workflow"},
        ),
        (
            lambda: TreeSearchPattern(
                generator_delegate=lambda _context: [{"candidate": "x"}],
                evaluator_delegate=lambda _context: 1.0,
                max_depth=1,
                branch_factor=1,
                beam_width=1,
            ),
            "Search one depth.",
            {"best_candidate", "final_output", "workflow"},
        ),
        (
            lambda: RAGPattern(
                reasoning_delegate=StaticMarkerAgent(marker="rag"),
                memory_store=None,
                write_back=False,
            ),
            "Reason with retrieval context.",
            {"retrieval", "reasoning", "final_output"},
        ),
    ],
)
def test_exported_patterns_run_output_contract(
    pattern_factory: Callable[[], object],
    prompt: str,
    expected_output_keys: set[str],
) -> None:
    pattern = pattern_factory()
    result = pattern.run(prompt)
    assert isinstance(result, ExecutionResult)
    assert expected_output_keys.issubset(set(result.output.keys()))


def test_plan_execute_workflow_output_contract_success_and_failure_paths() -> None:
    success_workflow = PlanExecutePattern(
        llm_client=SequenceLLMClient(
            response_texts=[
                json.dumps(
                    {
                        "steps": [
                            {
                                "step_id": "compute",
                                "instruction": "Count words in 'design research agents'.",
                                "success_criteria": "Return word count.",
                            }
                        ]
                    }
                ),
                '{"continue": true, "thought": "execute first step"}',
                "\n".join(
                    [
                        'stats = call_tool("text.word_count", {"text": "design research agents"})',
                        'final_output = {"result": stats["word_count"]}',
                    ]
                ),
            ]
        ),
        tool_runtime=Toolbox(),
        max_iterations=2,
    )
    success_result = success_workflow.run("Count words in design research agents.")
    assert success_result.success
    assert success_workflow.workflow is not None
    assert success_result.output["steps_executed"] == 1
    assert success_result.output["final_output"]["result"] == 3
    assert success_result.output["terminated_reason"] == "completed"
    assert success_result.metadata["runtime"]["resolved_mode"] == "plan_execute"

    failure_workflow = PlanExecutePattern(
        llm_client=SequenceLLMClient(response_texts=["invalid plan payload"]),
        tool_runtime=Toolbox(),
    )
    failure_result = failure_workflow.run("Count words in design research agents.")
    assert not failure_result.success
    assert failure_result.output["terminated_reason"] == "planner_invalid_json"
    assert failure_result.output["steps_executed"] == 0
    assert failure_result.output["step_results"] == []


def test_propose_and_critique_workflow_output_contract_success_and_failure_paths() -> None:
    success_workflow = ReflexionPattern(
        llm_client=SequenceLLMClient(
            response_texts=[
                "Draft v1",
                json.dumps(
                    {
                        "approved": True,
                        "feedback": "Looks good.",
                        "revision_goals": [],
                    }
                ),
            ]
        ),
        tool_runtime=Toolbox(),
        max_iterations=2,
    )
    success_result = success_workflow.run("Write a short design summary.")
    assert success_result.success
    assert success_workflow.workflow is not None
    assert success_result.output["approved"] is True
    assert success_result.output["terminated_reason"] == "approved"
    assert len(success_result.output["critique_iterations"]) == 1

    failure_workflow = ReflexionPattern(
        llm_client=SequenceLLMClient(
            response_texts=[
                "Draft v1",
                "invalid critique payload",
            ]
        ),
        tool_runtime=Toolbox(),
    )
    failure_result = failure_workflow.run("Write a short design summary.")
    assert not failure_result.success
    assert failure_result.output["terminated_reason"] == "critic_invalid_json"
    assert isinstance(failure_result.output["critique_iterations"], list)


def test_agent_routing_workflow_output_contract_success_and_failure_paths() -> None:
    success_workflow = AgentRoutingPattern(
        llm_client=SequenceLLMClient(response_texts=['{"tool_name":"alt_two","tool_input":{},"reason":"best fit"}']),
        tool_runtime=Toolbox(),
        alternatives={
            "alt_one": StaticMarkerAgent(marker="one"),
            "alt_two": StaticMarkerAgent(marker="two"),
        },
    )
    success_result = success_workflow.run("Route this request.")
    assert success_result.success
    assert success_workflow.workflow is not None
    assert success_result.output["agent_marker"] == "two"
    assert success_result.output["agent_routing_selected_alternative"] == "alt_two"
    assert success_result.metadata["agent_routing"]["selected_alternative"] == "alt_two"

    failure_workflow = AgentRoutingPattern(
        llm_client=SequenceLLMClient(
            response_texts=['{"tool_name":"unknown_alt","tool_input":{},"reason":"best fit"}']
        ),
        tool_runtime=Toolbox(),
        alternatives={"alt_one": StaticMarkerAgent(marker="one")},
    )
    failure_result = failure_workflow.run("Route this request.")
    assert not failure_result.success
    assert failure_result.output["terminated_reason"] == "routing_failure"
    assert failure_result.output["delegated_output"] == {}


def test_workflow_constructor_signatures_expose_new_default_kwargs() -> None:
    plan_params = inspect.signature(PlanExecutePattern.__init__).parameters
    assert "default_request_id_prefix" in plan_params
    assert "planner_system_prompt" in plan_params
    assert "planner_delegate" in plan_params
    assert "executor_delegate" in plan_params

    propose_params = inspect.signature(ReflexionPattern.__init__).parameters
    assert "proposer_user_prompt_template" in propose_params
    assert "default_dependencies" in propose_params
    assert "proposer_delegate" in propose_params
    assert "critic_delegate" in propose_params

    routing_params = inspect.signature(AgentRoutingPattern.__init__).parameters
    assert "router_system_prompt" in routing_params
    assert "default_request_id_prefix" in routing_params

    workflow_params = inspect.signature(Workflow.__init__).parameters
    assert "input_schema" in workflow_params
    assert "output_schema" in workflow_params
    assert "prompt_context_key" in workflow_params
    assert "default_execution_mode" in workflow_params
    assert "default_dependencies" in workflow_params


def test_workflow_factory_functions_are_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("design_research_agents.workflow.implementations")


def test_new_reasoning_and_networked_pattern_signatures_are_exposed() -> None:
    conversation_params = inspect.signature(ConversationPattern.__init__).parameters
    assert "llm_client_a" in conversation_params
    assert "llm_client_b" in conversation_params
    assert "speaker_a_delegate" in conversation_params
    assert "speaker_b_delegate" in conversation_params
    assert "max_turns" in conversation_params
    assert "speaker_a_user_prompt_template" in conversation_params

    networked_params = inspect.signature(NetworkedPattern.__init__).parameters
    assert "peers" in networked_params
    assert "max_rounds" in networked_params

    blackboard_params = inspect.signature(BlackboardPattern.__init__).parameters
    assert "stability_rounds" in blackboard_params

    tree_params = inspect.signature(TreeSearchPattern.__init__).parameters
    assert "generator_delegate" in tree_params
    assert "evaluator_delegate" in tree_params
    assert "beam_width" in tree_params

    rag_params = inspect.signature(RAGPattern.__init__).parameters
    assert "reasoning_delegate" in rag_params
    assert "memory_store" in rag_params
    assert "memory_top_k" in rag_params
