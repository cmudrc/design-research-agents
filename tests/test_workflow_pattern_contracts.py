"""Contract tests for reusable workflow pattern classes."""

from __future__ import annotations

import inspect
import json

from design_research_agents.tools import Toolbox
from design_research_agents.workflow import Workflow
from design_research_agents.workflow.implementations.agent_routing import RouterPattern
from design_research_agents.workflow.implementations.networked_blackboard import (
    BlackboardPattern,
    NetworkedPattern,
)
from design_research_agents.workflow.implementations.planner_executor_pattern import (
    PlannerExecutorPattern,
)
from design_research_agents.workflow.implementations.rag_reasoning import (
    RagReasoningPattern,
)
from design_research_agents.workflow.implementations.reflexion_pattern import (
    ReflexionPattern,
)
from design_research_agents.workflow.implementations.tree_search import (
    TreeSearchPattern,
)
from tests.helpers.workflow_stubs import SequenceLLMClient, StaticMarkerAgent


def test_plan_execute_workflow_output_contract_success_and_failure_paths() -> None:
    success_workflow = PlannerExecutorPattern(
        llm_client=SequenceLLMClient(
            response_texts=[
                json.dumps(
                    {
                        "steps": [
                            {
                                "step_id": "compute",
                                "instruction": "Compute 6 * 7.",
                                "success_criteria": "Return numeric result.",
                            }
                        ]
                    }
                ),
                "\n".join(
                    [
                        'calc = call_tool("calculator", {"expression": "6 * 7"})',
                        'final_output = {"result": calc["result"]}',
                    ]
                ),
            ]
        ),
        tool_runtime=Toolbox(),
        max_iterations=2,
    )
    success_result = success_workflow.run("Compute 6 * 7.")
    assert success_result.success
    assert success_result.output["steps_executed"] == 1
    assert success_result.output["final_output"]["result"] == 42.0
    assert success_result.output["terminated_reason"] == "completed"
    assert success_result.metadata["runtime"]["resolved_mode"] == "plan_execute"

    failure_workflow = PlannerExecutorPattern(
        llm_client=SequenceLLMClient(response_texts=["invalid plan payload"]),
        tool_runtime=Toolbox(),
    )
    failure_result = failure_workflow.run("Compute 6 * 7.")
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
    success_workflow = RouterPattern(
        llm_client=SequenceLLMClient(
            response_texts=['{"tool_names":["alt_two"],"reason":"best fit"}']
        ),
        tool_runtime=Toolbox(),
        alternatives={
            "alt_one": StaticMarkerAgent(marker="one"),
            "alt_two": StaticMarkerAgent(marker="two"),
        },
    )
    success_result = success_workflow.run("Route this request.")
    assert success_result.success
    assert success_result.output["agent_marker"] == "two"
    assert success_result.output["agent_routing_selected_alternative"] == "alt_two"
    assert success_result.metadata["agent_routing"]["selected_alternative"] == "alt_two"

    failure_workflow = RouterPattern(
        llm_client=SequenceLLMClient(
            response_texts=['{"tool_names":["unknown_alt"],"reason":"best fit"}']
        ),
        tool_runtime=Toolbox(),
        alternatives={"alt_one": StaticMarkerAgent(marker="one")},
    )
    failure_result = failure_workflow.run("Route this request.")
    assert not failure_result.success
    assert failure_result.output["terminated_reason"] == "routing_failure"
    assert failure_result.output["delegated_output"] == {}


def test_workflow_constructor_signatures_expose_new_default_kwargs() -> None:
    plan_params = inspect.signature(PlannerExecutorPattern.__init__).parameters
    assert "default_request_id_prefix" in plan_params
    assert "plan_execute_planner_system_prompt" in plan_params

    propose_params = inspect.signature(ReflexionPattern.__init__).parameters
    assert "propose_critic_proposer_user_prompt_template" in propose_params
    assert "default_dependencies" in propose_params

    routing_params = inspect.signature(RouterPattern.__init__).parameters
    assert "agent_routing_router_system_prompt" in routing_params
    assert "default_request_id_prefix" in routing_params

    workflow_params = inspect.signature(Workflow.__init__).parameters
    assert "input_mode" in workflow_params
    assert "input_schema" in workflow_params
    assert "prompt_context_key" in workflow_params
    assert "default_execution_mode" in workflow_params
    assert "default_dependencies" in workflow_params


def test_workflow_factory_functions_are_removed() -> None:
    from design_research_agents.workflow import implementations as workflow_impl

    removed_symbols = (
        "plan_execute_workflow",
        "propose_and_critique_workflow",
        "agent_routing_workflow",
        "mixed_agent_workflow",
        "pure_tool_workflow",
    )
    for symbol in removed_symbols:
        assert symbol not in workflow_impl.__all__


def test_new_reasoning_and_networked_pattern_signatures_are_exposed() -> None:
    networked_params = inspect.signature(NetworkedPattern.__init__).parameters
    assert "peers" in networked_params
    assert "max_rounds" in networked_params

    blackboard_params = inspect.signature(BlackboardPattern.__init__).parameters
    assert "stability_rounds" in blackboard_params

    tree_params = inspect.signature(TreeSearchPattern.__init__).parameters
    assert "generator_delegate" in tree_params
    assert "evaluator_delegate" in tree_params
    assert "beam_width" in tree_params

    rag_params = inspect.signature(RagReasoningPattern.__init__).parameters
    assert "reasoning_delegate" in rag_params
    assert "memory_store" in rag_params
    assert "memory_top_k" in rag_params
