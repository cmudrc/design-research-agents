"""Focused tests for delegate workflow expansion in diagrams."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from design_research_agents._contracts import Delegate, ExecutionResult
from design_research_agents.agent import DirectLLMCall
from design_research_agents.patterns import RouterDelegatePattern
from design_research_agents.tools import Toolbox
from design_research_agents.workflow import CompiledExecution, LogicStep, Workflow
from tests.helpers.workflow_stubs import SequenceLLMClient, StaticMarkerAgent


class _RecordingCompileDelegate(Delegate):
    """Delegate stub that records compile-call inputs for helper assertions."""

    def __init__(self) -> None:
        self.compile_calls: list[tuple[str, str | None, dict[str, object]]] = []
        self.workflow: Workflow | None = None

    def compile(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        captured_dependencies = dict(dependencies or {})
        self.compile_calls.append((prompt, request_id, captured_dependencies))
        workflow = Workflow(
            steps=[
                LogicStep(
                    step_id="recorded",
                    handler=lambda context: {"prompt": context["prompt"]},
                )
            ]
        )
        self.workflow = workflow
        return CompiledExecution(
            workflow=workflow,
            input=prompt,
            request_id=request_id or "recorded-request",
            dependencies=captured_dependencies,
            delegate_name="recording-compile-delegate",
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(
            output={},
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate": "recording-compile-delegate"},
        )


def test_router_delegate_diagrams_expose_alternative_delegate_branches() -> None:
    workflow = RouterDelegatePattern(
        llm_client=SequenceLLMClient(response_texts=['{"tool_name":"alt_two","tool_input":{},"reason":"best fit"}']),
        tool_runtime=Toolbox(),
        alternatives={
            "alt_one": StaticMarkerAgent(marker="one"),
            "alt_two": StaticMarkerAgent(marker="two"),
        },
    )

    compiled = workflow.compile("Route this request.")
    mermaid = compiled.workflow.to_mermaid(direction="LR")
    svg = compiled.workflow.to_svg(direction="LR")

    assert "agent_routing_selection" in mermaid
    assert "agent_routing_delegate_alt_one" in mermaid
    assert "agent_routing_delegate_alt_two" in mermaid
    assert "route=alt_one" in mermaid
    assert "route=alt_two" in mermaid

    assert "agent_routing_selection" in svg
    assert "agent_routing_delegate_alt_one" in svg
    assert "agent_routing_delegate_alt_two" in svg
    assert "route=alt_one" in svg
    assert "route=alt_two" in svg


def test_compiled_execution_diagram_helpers_delegate_to_workflow() -> None:
    workflow = RouterDelegatePattern(
        llm_client=SequenceLLMClient(response_texts=['{"tool_name":"alt_two","tool_input":{},"reason":"best fit"}']),
        tool_runtime=Toolbox(),
        alternatives={
            "alt_one": StaticMarkerAgent(marker="one"),
            "alt_two": StaticMarkerAgent(marker="two"),
        },
    )

    compiled = workflow.compile("Route this request.")

    assert compiled.to_mermaid(direction="LR") == compiled.workflow.to_mermaid(direction="LR")
    assert compiled.to_svg(direction="LR") == compiled.workflow.to_svg(direction="LR")


def test_delegate_default_diagram_helpers_compile_for_agents_and_patterns() -> None:
    agent = DirectLLMCall(llm_client=SequenceLLMClient(response_texts=["ignored"]))
    compiled_agent = agent.compile("Describe a design.")
    agent_mermaid = agent.compile_to_mermaid(direction="LR")
    agent_svg = agent.compile_to_svg(direction="LR")

    assert "prepare_request" in agent_mermaid
    assert "call_model" in agent_mermaid
    assert "finalize" in agent_mermaid
    assert "prepare_request" in agent_svg
    assert "call_model" in agent_svg
    assert "finalize" in agent_svg
    assert agent_mermaid == compiled_agent.workflow.to_mermaid(direction="LR")
    assert agent_svg == compiled_agent.workflow.to_svg(direction="LR")

    workflow = RouterDelegatePattern(
        llm_client=SequenceLLMClient(response_texts=['{"tool_name":"alt_two","tool_input":{},"reason":"best fit"}']),
        tool_runtime=Toolbox(),
        alternatives={
            "alt_one": StaticMarkerAgent(marker="one"),
            "alt_two": StaticMarkerAgent(marker="two"),
        },
    )
    compiled_pattern = workflow.compile("Route this request.")
    pattern_mermaid = workflow.compile_to_mermaid(direction="LR")
    pattern_svg = workflow.compile_to_svg(direction="LR")

    assert "agent_routing_selection" in pattern_mermaid
    assert "agent_routing_delegate_alt_two" in pattern_mermaid
    assert "route=alt_two" in pattern_mermaid
    assert "agent_routing_selection" in pattern_svg
    assert "agent_routing_delegate_alt_two" in pattern_svg
    assert "route=alt_two" in pattern_svg
    assert pattern_mermaid == compiled_pattern.workflow.to_mermaid(direction="LR")
    assert pattern_svg == compiled_pattern.workflow.to_svg(direction="LR")


def test_delegate_default_diagram_helpers_require_prior_compile() -> None:
    delegate = _RecordingCompileDelegate()

    with pytest.raises(RuntimeError, match="Compile the delegate once"):
        delegate.compile_to_mermaid(direction="LR")
    with pytest.raises(RuntimeError, match="Compile the delegate once"):
        delegate.compile_to_svg(direction="LR")

    assert delegate.compile_calls == []


def test_delegate_default_diagram_helpers_use_existing_workflow_without_recompile() -> None:
    delegate = _RecordingCompileDelegate()
    compiled = delegate.compile(
        "Map this workflow.",
        request_id="req-diagram",
        dependencies={"mode": "test"},
    )
    mermaid = delegate.compile_to_mermaid(direction="LR")
    svg = delegate.compile_to_svg(direction="LR")

    assert mermaid == compiled.workflow.to_mermaid(direction="LR")
    assert svg == compiled.workflow.to_svg(direction="LR")
    assert delegate.compile_calls == [
        ("Map this workflow.", "req-diagram", {"mode": "test"}),
    ]
