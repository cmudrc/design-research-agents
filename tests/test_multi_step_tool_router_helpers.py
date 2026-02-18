"""Tests for multi-step tool-router helper utilities."""

from __future__ import annotations

import json

from design_research_agents.agent.internal.multi_step_tool_router_helpers import (
    failure_result,
    normalize_output_dict,
    parse_tool_names,
    parse_tool_router_step_decision,
    resolve_selected_tool,
)
from design_research_agents.agent.internal.router_agent_helpers import ToolAlternative
from design_research_agents.contracts.llm import LLMResponse


def test_parse_tool_router_step_decision_normalizes_tool_call_payload() -> None:
    raw_text = json.dumps(
        {
            "action": "TOOL_CALL",
            "tool_names": ["search", "search", "fetch"],
            "tool_input": {"query": "hello"},
            "final_output": "done",
            "reason": "test",
        }
    )

    decision = parse_tool_router_step_decision(raw_text)

    assert decision is not None
    assert decision.action == "TOOL_CALL"
    assert decision.tool_names == ("search", "fetch")
    assert decision.tool_input == {"query": "hello"}
    assert decision.final_output == {"value": "done"}
    assert decision.reason == "test"
    assert decision.source == "model"


def test_parse_tool_router_helpers_cover_stop_and_selection_resolution() -> None:
    stop_decision = parse_tool_router_step_decision(
        json.dumps({"action": "STOP", "reason": "finished"})
    )
    assert stop_decision is not None
    assert stop_decision.action == "STOP"
    assert stop_decision.tool_names == ()

    alternatives = [
        ToolAlternative(tool_name="fetch", description="Fetch", input_schema={}),
        ToolAlternative(tool_name="search", description="Search", input_schema={}),
    ]
    assert resolve_selected_tool(alternatives=alternatives, tool_names=("search",)) == (
        "search",
        1,
    )
    assert resolve_selected_tool(alternatives=alternatives, tool_names=("unknown",)) is None


def test_parse_tool_names_normalize_output_and_failure_result() -> None:
    parsed = {"tool_names": ["lookup"]}
    assert parse_tool_names(parsed) == ("lookup",)
    assert parse_tool_names({"tool_name": "lookup"}) == ()
    assert normalize_output_dict({"ok": True}) == {"ok": True}
    assert normalize_output_dict(None) == {}
    assert normalize_output_dict(3) == {"value": 3}

    result = failure_result(
        error="boom",
        model_response=LLMResponse(text="response"),
        tool_results=[],
        request_id="req-1",
        dependencies={"beta": 2, "alpha": 1},
        metadata={"stage": "step"},
        output={"k": "v"},
    )
    assert not result.success
    assert result.output["error"] == "boom"
    assert result.metadata["request_id"] == "req-1"
    assert result.metadata["dependency_keys"] == ["alpha", "beta"]
