from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents._implementations._patterns import _router_delegate_pattern as routing_impl
from design_research_agents._implementations._shared._agent_internal import (
    AgentRoutingToolRuntimeAdapter,
)
from design_research_agents._implementations._shared._agent_internal import (
    _code_action_step_workflow_helpers as code_helpers,
)
from design_research_agents._runtime._patterns._run_context import (
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
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        return ToolResult(tool_name=tool_name, ok=True, result=dict(input))

    def close(self) -> None:
        return None

    def __enter__(self) -> _SingleToolRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


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
        code_helpers.assert_success_handler({"dependency_results": {"finalize": {"output": {"agent_result": "bad"}}}})

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
        == ""
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
        available_alternatives=("alpha",),
        workflow_payload={"success": False},
        workflow_artifacts=(),
    )
    assert failure.success is False
    assert failure.output["terminated_reason"] == "routing_failure"
    assert failure.output["details"]["available_alternatives"] == ["alpha"]
    assert failure.metadata["stage"] == "agent_routing_selection"

    runtime = _SingleToolRuntime()
    assert runtime.list_tools()[0].name == "sum"


def test_agent_routing_runtime_adapter_supports_context_manager() -> None:
    class _RouteTarget:
        def run(
            self,
            prompt: str,
            *,
            request_id: str | None = None,
            dependencies: Mapping[str, object] | None = None,
        ) -> ExecutionResult:
            del prompt, request_id, dependencies
            return ExecutionResult(success=True, output={"selected": True})

    adapter = AgentRoutingToolRuntimeAdapter(alternatives={"alpha": _RouteTarget()})

    with adapter as entered:
        assert entered is adapter
        tool_names = [spec.name for spec in entered.list_tools()]
        result = entered.invoke("alpha", {}, request_id="req", dependencies={})

    assert tool_names == ["alpha"]
    assert result.ok is True
    assert result.result["selected_alternative"] == "alpha"
