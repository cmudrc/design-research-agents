"""Focused tests for delegate workflow expansion in diagrams."""

from __future__ import annotations

from design_research_agents.patterns import RouterDelegatePattern
from design_research_agents.tools import Toolbox
from tests.helpers.workflow_stubs import SequenceLLMClient, StaticMarkerAgent


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
