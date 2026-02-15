"""Tests for AgentRuntime modes and workflow orchestrators."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import pytest

from design_research_agents.agent import AgentRuntime, MultiStepAgent, RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents.orchestrator import DagOrchestrator, SequentialOrchestrator
from design_research_agents.tools import BaseToolRuntime


class _SequenceLLMClient:
    """Deterministic LLM stub that returns configured responses in order."""

    def __init__(self, *, response_texts: list[str]) -> None:
        self._responses = list(response_texts)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages, params
        if not self._responses:
            raise AssertionError("No more stubbed responses available.")
        return LLMResponse(
            model=model,
            text=self._responses.pop(0),
            provider="test-sequence",
            latency_ms=4,
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        response = self.chat(messages, model=model, params=params)
        yield LLMStreamEvent(kind="delta", delta_text=response.text)
        yield LLMStreamEvent(kind="completed", response=response)

    def generate(self, request: LLMRequest) -> LLMResponse:
        return self.chat(
            list(request.messages),
            model=request.model or self.default_model(),
            params=LLMChatParams(),
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        response = self.generate(request)
        yield LLMDelta(text_delta=response.text)

    def default_model(self) -> str:
        return "test-model"


class _StaticAgent(Agent):
    """Simple deterministic agent used for triage tests."""

    def __init__(self, *, marker: str) -> None:
        self._marker = marker

    def run(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del input, request_id, dependencies
        return AgentResult(
            output={"agent_marker": self._marker},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"agent": self._marker},
        )

    def run_stream(
        self,
        input: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator:
        del input, request_id, dependencies
        raise NotImplementedError


def test_agent_runtime_react_mode_aliases_multi_step_agent() -> None:
    response_texts = [
        '{"continue": true, "reason": "start"}',
        "\n".join(
            [
                'calc = call_tool("calculator_tool", {"expression": "6 * 7"})',
                'final_output = {"result": calc["result"]}',
            ]
        ),
        '{"continue": false, "reason": "done"}',
    ]

    llm_runtime = _SequenceLLMClient(response_texts=response_texts)
    llm_direct = _SequenceLLMClient(response_texts=list(response_texts))
    tool_runtime = BaseToolRuntime()

    runtime_agent = AgentRuntime(
        llm_client=llm_runtime,
        tool_runtime=tool_runtime,
        mode="react",
        controls=RuntimeControls(max_steps=3),
    )
    direct_agent = MultiStepAgent(
        llm_client=llm_direct,
        tool_runtime=tool_runtime,
        max_steps=3,
    )

    runtime_result = runtime_agent.run("Compute 6 * 7.")
    direct_result = direct_agent.run("Compute 6 * 7.")

    assert runtime_result.success == direct_result.success
    assert runtime_result.output["final_output"] == direct_result.output["final_output"]
    assert runtime_result.output["terminated_reason"] == direct_result.output["terminated_reason"]
    assert runtime_result.metadata["runtime"]["resolved_mode"] == "multi_step_agent"


def test_agent_runtime_plan_execute_mode_runs_planner_then_executor() -> None:
    llm_client = _SequenceLLMClient(
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
                    'calc = call_tool("calculator_tool", {"expression": "6 * 7"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm_client,
        tool_runtime=BaseToolRuntime(),
        mode="plan_execute",
        controls=RuntimeControls(max_iterations=2),
    )

    result = runtime.run("Compute 6 * 7.")

    assert result.success
    assert result.output["steps_executed"] == 1
    assert result.output["final_output"]["result"] == 42.0
    assert result.metadata["runtime"]["resolved_mode"] == "plan_execute"


def test_agent_runtime_propose_critic_stops_on_approval() -> None:
    llm_client = _SequenceLLMClient(
        response_texts=[
            "Draft v1",
            json.dumps(
                {
                    "approved": False,
                    "feedback": "Add more detail.",
                    "revision_goals": ["expand rationale"],
                }
            ),
            "Draft v2 with more detail",
            json.dumps(
                {
                    "approved": True,
                    "feedback": "Looks good.",
                    "revision_goals": [],
                }
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm_client,
        tool_runtime=BaseToolRuntime(),
        mode="propose_critic",
        controls=RuntimeControls(max_iterations=3),
    )

    result = runtime.run("Write a short design summary.")

    assert result.success
    assert result.output["approved"] is True
    assert result.output["terminated_reason"] == "approved"
    assert len(result.output["critique_iterations"]) == 2


def test_agent_runtime_triage_selects_and_executes_named_alternative() -> None:
    llm_client = _SequenceLLMClient(
        response_texts=[
            '{"selection": "alt_two", "reason": "best fit"}',
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm_client,
        tool_runtime=BaseToolRuntime(),
        mode="triage",
        triage_alternatives={
            "alt_one": _StaticAgent(marker="one"),
            "alt_two": _StaticAgent(marker="two"),
        },
    )

    result = runtime.run("Handle this request.")

    assert result.success
    assert result.output["agent_marker"] == "two"
    assert result.output["triage_selected_alternative"] == "alt_two"
    assert result.metadata["triage"]["selected_alternative"] == "alt_two"


def test_agent_runtime_stream_emits_delta_then_completed() -> None:
    llm_client = _SequenceLLMClient(
        response_texts=[
            json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "s1",
                            "instruction": "Compute 6 * 7.",
                            "success_criteria": "Return numeric result.",
                        }
                    ]
                }
            ),
            "\n".join(
                [
                    'calc = call_tool("calculator_tool", {"expression": "6 * 7"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm_client,
        tool_runtime=BaseToolRuntime(),
        mode="plan_execute",
    )

    events = list(runtime.run_stream("Compute 6 * 7."))

    assert [event.kind for event in events] == ["delta", "completed"]
    assert events[1].result is not None
    assert events[1].result.success


@dataclass
class _TestNode:
    node_id: str
    dependencies: tuple[str, ...]
    run_fn: Callable[[Mapping[str, object]], Mapping[str, object]]
    input_schema: Mapping[str, object]
    output_schema: Mapping[str, object]
    route_map: Mapping[str, tuple[str, ...]] | None = None

    def run(self, context: Mapping[str, object]) -> Mapping[str, object]:
        return self.run_fn(context)


_NODE_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["dependency_results"],
    "properties": {
        "dependency_results": {"type": "object"},
    },
    "additionalProperties": True,
}

_NODE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["value"],
    "properties": {
        "value": {"type": "integer"},
    },
    "additionalProperties": False,
}


def test_sequential_orchestrator_runs_dependency_order() -> None:
    orchestrator = SequentialOrchestrator()
    nodes = [
        _TestNode(
            node_id="a",
            dependencies=(),
            run_fn=lambda ctx: {"value": 1},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="b",
            dependencies=("a",),
            run_fn=lambda ctx: {
                "value": int(ctx["dependency_results"]["a"]["output"]["value"]) + 1,
            },
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
    ]

    result = orchestrator.run(nodes)

    assert result.success
    assert result.execution_order == ["a", "b"]
    assert result.node_results["b"].output["value"] == 2


def test_sequential_orchestrator_failure_policy_skip_dependents() -> None:
    orchestrator = SequentialOrchestrator()
    nodes = [
        _TestNode(
            node_id="a",
            dependencies=(),
            run_fn=lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="b",
            dependencies=("a",),
            run_fn=lambda ctx: {"value": 2},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
    ]

    result = orchestrator.run(nodes, failure_policy="skip_dependents")

    assert not result.success
    assert result.node_results["a"].status == "failed"
    assert result.node_results["b"].status == "skipped"
    assert result.node_results["b"].error == "skipped_upstream_failure"


def test_sequential_orchestrator_failure_policy_propagate_failed_state() -> None:
    orchestrator = SequentialOrchestrator()
    nodes = [
        _TestNode(
            node_id="a",
            dependencies=(),
            run_fn=lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="b",
            dependencies=("a",),
            run_fn=lambda ctx: {"value": 2},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
    ]

    result = orchestrator.run(nodes, failure_policy="propagate_failed_state")

    assert not result.success
    assert result.node_results["a"].status == "failed"
    assert result.node_results["b"].status == "completed"


def test_dag_orchestrator_has_deterministic_topological_order() -> None:
    orchestrator = DagOrchestrator()
    nodes = [
        _TestNode(
            node_id="a",
            dependencies=(),
            run_fn=lambda ctx: {"value": 1},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="c",
            dependencies=("a",),
            run_fn=lambda ctx: {"value": 3},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="b",
            dependencies=("a",),
            run_fn=lambda ctx: {"value": 2},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="d",
            dependencies=("b", "c"),
            run_fn=lambda ctx: {"value": 4},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
    ]

    result = orchestrator.run(nodes)

    assert result.success
    assert result.execution_order == ["a", "b", "c", "d"]


def test_dag_orchestrator_detects_cycle_with_clear_error() -> None:
    orchestrator = DagOrchestrator()
    nodes = [
        _TestNode(
            node_id="a",
            dependencies=("b",),
            run_fn=lambda ctx: {"value": 1},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="b",
            dependencies=("a",),
            run_fn=lambda ctx: {"value": 2},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        orchestrator.run(nodes)


def test_dag_orchestrator_single_router_fan_out_skips_non_selected_branch() -> None:
    orchestrator = DagOrchestrator()

    router_output_schema = {
        "type": "object",
        "required": ["value", "route"],
        "properties": {
            "value": {"type": "integer"},
            "route": {"type": "string"},
        },
        "additionalProperties": False,
    }

    nodes = [
        _TestNode(
            node_id="router",
            dependencies=(),
            run_fn=lambda ctx: {"value": 1, "route": "left"},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=router_output_schema,
            route_map={"left": ("left_node",), "right": ("right_node",)},
        ),
        _TestNode(
            node_id="left_node",
            dependencies=("router",),
            run_fn=lambda ctx: {"value": 2},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
        _TestNode(
            node_id="right_node",
            dependencies=("router",),
            run_fn=lambda ctx: {"value": 3},
            input_schema=_NODE_INPUT_SCHEMA,
            output_schema=_NODE_OUTPUT_SCHEMA,
        ),
    ]

    result = orchestrator.run(nodes, failure_policy="propagate_failed_state")

    assert result.success
    assert result.node_results["left_node"].status == "completed"
    assert result.node_results["right_node"].status == "skipped"
    assert result.node_results["right_node"].error == "skipped_branch_not_selected"
