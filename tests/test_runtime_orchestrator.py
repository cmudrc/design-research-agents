"""Tests for AgentRuntime modes and WorkflowRuntime orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence

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
from design_research_agents.contracts.orchestrator import AgentStep, LogicStep, ToolStep
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.orchestrator import WorkflowRuntime
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
    """Simple deterministic agent used for triage and workflow tests."""

    def __init__(self, *, marker: str) -> None:
        self._marker = marker

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        del prompt, request_id, dependencies
        return AgentResult(
            output={"agent_marker": self._marker},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"agent": self._marker},
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator:
        del prompt, request_id, dependencies
        raise NotImplementedError


class _StubToolRuntime(ToolRuntime):
    """Small in-memory tool runtime for workflow tests."""

    def __init__(
        self,
        *,
        handlers: Mapping[str, Callable[[Mapping[str, object]], Mapping[str, object]]],
    ) -> None:
        self._handlers = dict(handlers)
        self._specs = {
            name: ToolSpec(
                name=name,
                description=f"Test tool '{name}'.",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            )
            for name in handlers
        }

    def list_tools(self) -> Sequence[ToolSpec]:
        return tuple(self._specs.values())

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                result={},
                error=f"Tool '{tool_name}' is not registered.",
            )

        try:
            output = dict(handler(input_dict))
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                result={},
                error=str(exc),
            )

        return ToolResult(tool_name=tool_name, ok=True, result=output)


def test_agent_runtime_react_mode_aliases_multi_step_agent() -> None:
    response_texts = [
        '{"continue": true, "reason": "start"}',
        "\n".join(
            [
                'calc = call_tool("calculator", {"expression": "6 * 7"})',
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
                    'calc = call_tool("calculator", {"expression": "6 * 7"})',
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
                    'calc = call_tool("calculator", {"expression": "6 * 7"})',
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


def test_workflow_runtime_sequential_runs_dependency_order() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="a", handler=lambda ctx: {"value": 1}),
        LogicStep(
            step_id="b",
            dependencies=("a",),
            handler=lambda ctx: {
                "value": int(ctx["dependency_results"]["a"]["output"]["value"]) + 1,
            },
        ),
    ]

    result = workflow.run(steps, execution_mode="sequential")

    assert result.success
    assert result.execution_order == ["a", "b"]
    assert result.step_results["b"].output["value"] == 2


def test_workflow_runtime_sequential_raises_for_unresolved_dependencies() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
        LogicStep(step_id="a", handler=lambda ctx: {"value": 1}),
    ]

    with pytest.raises(ValueError, match="cannot run before dependencies are resolved"):
        workflow.run(steps, execution_mode="sequential")


def test_workflow_runtime_dag_has_deterministic_topological_order() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="a", handler=lambda ctx: {"value": 1}),
        LogicStep(step_id="c", dependencies=("a",), handler=lambda ctx: {"value": 3}),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
        LogicStep(
            step_id="d",
            dependencies=("b", "c"),
            handler=lambda ctx: {"value": 4},
        ),
    ]

    result = workflow.run(steps, execution_mode="dag")

    assert result.success
    assert result.execution_order == ["a", "b", "c", "d"]


def test_workflow_runtime_dag_detects_cycle_with_clear_error() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(step_id="a", dependencies=("b",), handler=lambda ctx: {"value": 1}),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
    ]

    with pytest.raises(ValueError, match="Cycle detected"):
        workflow.run(steps, execution_mode="dag")


def test_workflow_runtime_failure_policy_skip_dependents() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(
            step_id="a",
            handler=lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="skip_dependents",
    )

    assert not result.success
    assert result.step_results["a"].status == "failed"
    assert result.step_results["b"].status == "skipped"
    assert result.step_results["b"].error == "skipped_upstream_failure"


def test_workflow_runtime_failure_policy_propagate_failed_state() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(
            step_id="a",
            handler=lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
        LogicStep(step_id="b", dependencies=("a",), handler=lambda ctx: {"value": 2}),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="propagate_failed_state",
    )

    assert not result.success
    assert result.step_results["a"].status == "failed"
    assert result.step_results["b"].status == "completed"


def test_workflow_runtime_route_branching_skips_non_selected_branch() -> None:
    workflow = WorkflowRuntime()
    steps = [
        LogicStep(
            step_id="router",
            handler=lambda ctx: {"route": "left"},
            route_map={"left": ("left_step",), "right": ("right_step",)},
        ),
        LogicStep(step_id="left_step", dependencies=("router",), handler=lambda ctx: {"value": 1}),
        LogicStep(
            step_id="right_step",
            dependencies=("router",),
            handler=lambda ctx: {"value": 2},
        ),
    ]

    result = workflow.run(steps, execution_mode="dag", failure_policy="propagate_failed_state")

    assert result.success
    assert result.step_results["left_step"].status == "completed"
    assert result.step_results["right_step"].status == "skipped"
    assert result.step_results["right_step"].error == "skipped_branch_not_selected"


def test_workflow_runtime_tool_step_returns_serialized_tool_result() -> None:
    tool_runtime = _StubToolRuntime(
        handlers={
            "adder_tool": lambda payload: {
                "sum": float(payload.get("a", 0)) + float(payload.get("b", 0))
            }
        }
    )
    workflow = WorkflowRuntime(tool_runtime=tool_runtime)
    steps = [
        ToolStep(step_id="add", tool_name="adder_tool", input_data={"a": 40, "b": 2}),
    ]

    result = workflow.run(steps, execution_mode="sequential")

    assert result.success
    step_output = result.step_results["add"].output
    assert step_output["tool_name"] == "adder_tool"
    assert step_output["ok"] is True
    assert step_output["result"]["sum"] == 42.0


def test_workflow_runtime_agent_step_returns_serialized_agent_result() -> None:
    workflow = WorkflowRuntime(agents={"math_agent": _StaticAgent(marker="math")})
    steps = [
        AgentStep(step_id="delegate", agent_name="math_agent", prompt="Solve this."),
    ]

    result = workflow.run(steps, execution_mode="sequential")

    assert result.success
    step_output = result.step_results["delegate"].output
    assert step_output["success"] is True
    assert step_output["output"]["agent_marker"] == "math"


def test_workflow_runtime_mixed_pipeline_supports_logic_agent_and_tool_steps() -> None:
    tool_runtime = _StubToolRuntime(
        handlers={
            "text_length_tool": lambda payload: {
                "length": len(str(payload.get("text", ""))),
            }
        }
    )
    workflow = WorkflowRuntime(
        tool_runtime=tool_runtime,
        agents={"writer_agent": _StaticAgent(marker="proposal")},
    )
    steps = [
        LogicStep(
            step_id="router",
            handler=lambda ctx: {"route": "agent_path"},
            route_map={"agent_path": ("delegate",), "other_path": ("unused",)},
        ),
        AgentStep(
            step_id="delegate",
            agent_name="writer_agent",
            dependencies=("router",),
            prompt_builder=lambda ctx: "Write a proposal.",
        ),
        LogicStep(
            step_id="unused",
            dependencies=("router",),
            handler=lambda ctx: {"value": "should skip"},
        ),
        ToolStep(
            step_id="measure",
            tool_name="text_length_tool",
            dependencies=("delegate",),
            input_builder=lambda ctx: {
                "text": ctx["dependency_results"]["delegate"]["output"]["output"]["agent_marker"]
            },
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("measure",),
            handler=lambda ctx: {
                "length": ctx["dependency_results"]["measure"]["output"]["result"]["length"]
            },
        ),
    ]

    result = workflow.run(steps, execution_mode="dag")

    assert result.success
    assert result.step_results["unused"].status == "skipped"
    assert result.step_results["unused"].error == "skipped_branch_not_selected"
    assert result.step_results["finalize"].output["length"] == len("proposal")


def test_workflow_runtime_unknown_bindings_fail_with_stage_metadata() -> None:
    tool_runtime = _StubToolRuntime(handlers={"known_tool": lambda payload: {"ok": True}})
    workflow = WorkflowRuntime(
        tool_runtime=tool_runtime,
        agents={"known_agent": _StaticAgent(marker="ok")},
    )
    steps = [
        ToolStep(step_id="missing_tool", tool_name="unknown_tool"),
        AgentStep(step_id="missing_agent", agent_name="unknown_agent", prompt="Do work."),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="propagate_failed_state",
    )

    assert not result.success
    assert result.step_results["missing_tool"].status == "failed"
    assert result.step_results["missing_tool"].metadata["stage"] == "tool_binding"
    assert result.step_results["missing_agent"].status == "failed"
    assert result.step_results["missing_agent"].metadata["stage"] == "agent_binding"
