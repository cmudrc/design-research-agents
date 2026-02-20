from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.implementations.patterns import agent_routing as routing_impl
from design_research_agents.implementations.shared.agent_internal import (
    code_action_step_workflow_helpers as code_helpers,
)
from design_research_agents.implementations.shared.agent_internal import (
    router_agent_helpers as route_helpers,
)
from design_research_agents.implementations.shared.workflow_internal.pattern_runtime import (
    WorkflowBudgetTracker,
)


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
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        return ToolResult(tool_name=tool_name, ok=True, result=dict(input_dict))


def _tool_spec(name: str, description: str = "tool") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={"type": "object", "additionalProperties": True},
        output_schema={"type": "object", "additionalProperties": True},
    )


def test_code_action_step_workflow_helpers_cover_success_and_failure_branches() -> None:
    finalized_success = {
        "dependency_results": {
            "finalize": {
                "output": {
                    "agent_result": ExecutionResult(success=True),
                }
            }
        }
    }
    assert code_helpers.assert_success_handler(finalized_success) == {"ok": True}

    with pytest.raises(TypeError, match="Finalize step"):
        code_helpers.assert_success_handler(
            {"dependency_results": {"finalize": {"output": {"agent_result": "bad"}}}}
        )

    with pytest.raises(ValueError, match="boom"):
        code_helpers.assert_success_handler(
            {
                "dependency_results": {
                    "finalize": {
                        "output": {
                            "agent_result": ExecutionResult(
                                success=False,
                                output={"error": "boom"},
                            )
                        }
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="code action-step execution failed"):
        code_helpers.assert_success_handler(
            {
                "dependency_results": {
                    "finalize": {
                        "output": {
                            "agent_result": ExecutionResult(
                                success=False,
                                output={"error": object()},
                            )
                        }
                    }
                }
            }
        )

    assert code_helpers.dependency_output(context={}, step_id="x") == {}
    assert code_helpers.dependency_output(
        context={"dependency_results": {"x": {"output": {"k": 1}}}},
        step_id="x",
    ) == {"k": 1}
    assert code_helpers.mapping_or_empty({"k": 1}) == {"k": 1}
    assert code_helpers.mapping_or_empty("bad") == {}
    assert code_helpers.int_or_default(True, default=9) == 1
    assert code_helpers.int_or_default(4, default=9) == 4
    assert code_helpers.int_or_default("5", default=9) == 5
    assert code_helpers.int_or_default("bad", default=9) == 9
    assert code_helpers.int_or_default(None, default=9) == 9
    response = LLMResponse(text="ok")
    assert code_helpers.llm_response_or_none(response) is response
    assert code_helpers.llm_response_or_none("bad") is None


def test_router_agent_helpers_cover_prompt_parse_and_resolution_paths() -> None:
    runtime_specs = {
        "sum": _tool_spec("sum", "Add"),
        "text.word_count": _tool_spec("text.word_count", "Count words"),
    }
    compiled = route_helpers.compile_runtime_alternatives(tool_specs=runtime_specs)
    assert [item.tool_name for item in compiled] == ["sum", "text.word_count"]

    filtered = route_helpers.compile_runtime_alternatives(
        tool_specs=runtime_specs,
        allowed_route_names=("sum",),
    )
    assert [item.tool_name for item in filtered] == ["sum"]

    extracted = route_helpers.extract_alternatives(
        runtime_specs=runtime_specs,
        compiled_runtime_alternatives=filtered,
    )
    assert extracted[0].tool_name == "sum"
    assert extracted[0] is not filtered[0]

    with pytest.raises(ValueError, match="requires at least one tool"):
        route_helpers.extract_alternatives(
            runtime_specs=runtime_specs,
            compiled_runtime_alternatives=(),
        )

    assert (
        route_helpers.resolve_allowed_route_names(
            runtime_specs=runtime_specs,
            allowed_routes=None,
        )
        is None
    )
    assert route_helpers.resolve_allowed_route_names(
        runtime_specs=runtime_specs,
        allowed_routes=["sum", " sum ", "text.word_count", "missing"],
    ) == ("sum", "text.word_count")
    with pytest.raises(ValueError, match="did not match"):
        route_helpers.resolve_allowed_route_names(
            runtime_specs=runtime_specs,
            allowed_routes=["missing"],
        )

    routes_text = route_helpers.build_routes_text(
        alternatives=[
            route_helpers.ToolAlternative(
                tool_name="sum",
                description="",
                input_schema={"type": "object"},
            )
        ]
    )
    assert "(none)" in routes_text

    route_prompt = route_helpers.build_route_prompt(
        prompt="Compute 2 + 3",
        routes_block="- tool_name: sum",
        prompt_template="Routes:\n$routes_block\nPrompt: $user_prompt",
    )
    assert "Compute 2 + 3" in route_prompt

    schema = route_helpers.route_response_schema(alternatives=filtered)
    tool_name_enum = schema["properties"]["tool_names"]["items"]["enum"]
    assert tool_name_enum == ["sum"]

    assert route_helpers.parse_route_response("not-json") is None
    parsed = route_helpers.parse_route_response(
        '{"tool_names": [" missing ", "sum", "sum"], "reason": "because"}'
    )
    assert parsed is not None
    assert parsed.tool_names == ("missing", "sum")

    assert route_helpers._parse_tool_names({"tool_names": "sum"}) == ()
    assert route_helpers._parse_tool_names({"tool_names": [1, " ", "sum"]}) == ("sum",)

    resolved_route = route_helpers.resolve_model_route(
        parsed_route=parsed,
        alternatives=filtered,
    )
    assert resolved_route is not None
    assert resolved_route[0].tool_name == "sum"
    assert resolved_route[2] == "because"
    assert route_helpers.resolve_model_route(parsed_route=None, alternatives=filtered) is None
    assert (
        route_helpers.resolve_model_route(
            parsed_route=route_helpers.ParsedRoute(tool_names=("missing",), reason=None),
            alternatives=filtered,
        )
        is None
    )

    assert route_helpers.resolve_tool_input(
        tool_name="sum",
        input_payload={"tool_input": {"a": 1}},
    ) == {"a": 1}

    known_input = route_helpers.resolve_tool_input(
        tool_name="calculator",
        input_payload={"prompt": "Compute 4 + 5"},
    )
    assert isinstance(known_input.get("expression"), str)
    assert "+" in str(known_input["expression"])

    assert route_helpers.resolve_tool_input(
        tool_name="unknown",
        input_payload={"prompt": "fallback"},
    ) == {"prompt": "fallback", "request": "fallback"}

    invalid_response = LLMResponse(text="bad")
    failure = route_helpers.routing_failure_result(
        error="bad route",
        llm_response=invalid_response,
        request_id="req-1",
        dependencies={"dep": 1},
        alternatives=list(filtered),
        parsed_route=route_helpers.ParsedRoute(tool_names=("sum",), reason="x"),
    )
    assert failure.success is False
    assert failure.output["error"] == "bad route"
    assert failure.metadata["routing"]["parsed_route"]["tool_names"] == ["sum"]


def test_agent_routing_helper_extractors_cover_selection_shapes() -> None:
    assert routing_impl._extract_selection_output({}) is None
    assert routing_impl._extract_selection_output({"dependency_results": {}}) is None
    assert routing_impl._extract_selection_output(
        {
            "dependency_results": {
                "agent_routing_selection": {
                    "output": {
                        "status": "selected",
                        "selected_name": "alpha",
                    }
                }
            }
        }
    ) == {"status": "selected", "selected_name": "alpha"}

    assert routing_impl._extract_selected_name_from_router_output({}) == ""
    assert (
        routing_impl._extract_selected_name_from_router_output(
            {"step_outputs": [{"action": "TOOL_CALL", "tool_name": "alpha"}]}
        )
        == "alpha"
    )
    assert (
        routing_impl._extract_selected_name_from_router_output(
            {
                "step_outputs": [
                    {"action": "STOP"},
                    {
                        "action": "TOOL_CALL",
                        "tool_name": " ",
                        "tool_names": [1, " beta "],
                    },
                ]
            }
        )
        == "beta"
    )

    budget_tracker = WorkflowBudgetTracker()
    router_result = ExecutionResult(
        success=False,
        output={},
        tool_results=[],
        model_response=None,
        metadata={"routing": {"source": "model"}},
    )
    failure = routing_impl._build_routing_failure_result(
        error="route failed",
        request_id="req",
        dependencies={"dep": 1},
        router_result=router_result,
        budget_tracker=budget_tracker,
        stage="agent_routing_selection",
        terminated_reason="routing_failure",
        workflow_payload={"success": False},
        workflow_artifacts=(),
    )
    assert failure.success is False
    assert failure.output["terminated_reason"] == "routing_failure"
    assert failure.metadata["stage"] == "agent_routing_selection"

    runtime = _SingleToolRuntime()
    assert runtime.list_tools()[0].name == "sum"
