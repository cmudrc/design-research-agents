"""Focused branch-coverage tests for single-step workflow helper paths."""

from __future__ import annotations

import pytest

import design_research_agents.agent as agent_facade
from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.workflow import WorkflowStepResult
from design_research_agents.implementations.agents import (
    single_step_direct_llm_agent as direct_module,
)
from design_research_agents.implementations.agents import (
    single_step_json_tool_calling_agent as json_module,
)
from design_research_agents.implementations.shared.agent_internal import (
    single_step_code_workflow_helpers as code_helpers,
)


def _step_result(
    *,
    success: bool,
    output: object,
    error: str | None = None,
) -> WorkflowStepResult:
    return WorkflowStepResult(
        step_id="step",
        status="completed" if success else "failed",
        success=success,
        output=output,  # type: ignore[arg-type]
        error=error,
    )


def test_agent_facade_dir_and_missing_attr_paths() -> None:
    names = dir(agent_facade)
    assert "SingleStepDirectLLMAgent" in names
    with pytest.raises(AttributeError):
        _ = agent_facade.DefinitelyMissingAgentSymbol  # type: ignore[attr-defined]


def test_code_helper_assert_success_paths() -> None:
    with pytest.raises(TypeError, match="ExecutionResult"):
        code_helpers.assert_success_handler(context={})

    failed_result = ExecutionResult(success=False, output={"error": "boom"})
    failed_context = {
        "dependency_results": {
            "finalize": {
                "output": {
                    "agent_result": failed_result,
                }
            }
        }
    }
    with pytest.raises(ValueError, match="boom"):
        code_helpers.assert_success_handler(context=failed_context)

    success_result = ExecutionResult(success=True, output={})
    success_context = {
        "dependency_results": {
            "finalize": {
                "output": {
                    "agent_result": success_result,
                }
            }
        }
    }
    assert code_helpers.assert_success_handler(context=success_context) == {"ok": True}


def test_code_helper_dependency_and_scalar_coercions() -> None:
    assert code_helpers.dependency_output(context={}, step_id="x") == {}
    assert (
        code_helpers.dependency_output(
            context={"dependency_results": {"x": "not-a-step"}},  # type: ignore[dict-item]
            step_id="x",
        )
        == {}
    )
    assert (
        code_helpers.dependency_output(
            context={"dependency_results": {"x": {"output": "not-a-mapping"}}},
            step_id="x",
        )
        == {}
    )
    assert code_helpers.dependency_output(
        context={"dependency_results": {"x": {"output": {"ok": 1}}}},
        step_id="x",
    ) == {"ok": 1}

    assert code_helpers.mapping_or_empty("nope") == {}
    assert code_helpers.int_or_default(True, default=7) == 1
    assert code_helpers.int_or_default(5, default=7) == 5
    assert code_helpers.int_or_default("12", default=7) == 12
    assert code_helpers.int_or_default("abc", default=7) == 7
    assert code_helpers.int_or_default(object(), default=7) == 7
    assert code_helpers.llm_response_or_none(object()) is None
    response = LLMResponse(text="ok")
    assert code_helpers.llm_response_or_none(response) is response


def test_direct_module_helper_failure_paths() -> None:
    with pytest.raises(ValueError, match="bad"):
        direct_module._raise_workflow_failure(
            ExecutionResult(
                success=False,
                output={},
                step_results={"a": _step_result(success=False, output={}, error="bad")},
                execution_order=["a"],
            )
        )

    with pytest.raises(RuntimeError, match="step 'a' failed"):
        direct_module._raise_workflow_failure(
            ExecutionResult(
                success=False,
                output={},
                step_results={"a": _step_result(success=False, output={}, error=None)},
                execution_order=["a"],
            )
        )

    with pytest.raises(RuntimeError, match="execution failed"):
        direct_module._raise_workflow_failure(
            ExecutionResult(
                success=False,
                output={},
                step_results={"a": object()},
                execution_order=["a"],
            )
        )

    assert direct_module._dependency_output(context={}, step_id="x") == {}
    assert (
        direct_module._dependency_output(
            context={"dependency_results": {"x": "not-a-step"}},  # type: ignore[dict-item]
            step_id="x",
        )
        == {}
    )
    assert direct_module._dependency_output(
        context={"dependency_results": {"x": {"output": {"ok": 1}}}},
        step_id="x",
    ) == {"ok": 1}
    assert direct_module._int_or_default("42", default=3) == 42
    assert direct_module._int_or_default("x", default=3) == 3


@pytest.mark.parametrize(
    ("context", "message"),
    [
        ({}, "Missing dependency_results"),
        ({"dependency_results": {}}, "Missing select_tool result"),
        ({"dependency_results": {"select_tool": {}}}, "Invalid select_tool output"),
    ],
)
def test_json_invalid_selection_handler_guards(
    *,
    context: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        json_module._invalid_selection_handler(context)


def test_json_helper_builders_and_failure_extractors() -> None:
    builder = json_module._build_tool_input_builder(expected_tool_name="calculator")
    assert builder({}) == {}
    assert builder({"dependency_results": {"select_tool": {}}}) == {}
    assert builder({"dependency_results": {"select_tool": {"output": {}}}}) == {}
    assert (
        builder(
            {
                "dependency_results": {
                    "select_tool": {
                        "output": {
                            "tool_name": "search",
                            "tool_input": {"q": "hi"},
                        }
                    }
                }
            }
        )
        == {}
    )
    assert builder(
        {
            "dependency_results": {
                "select_tool": {
                    "output": {
                        "tool_name": "calculator",
                        "tool_input": {"expression": "1+1"},
                    }
                }
            }
        }
    ) == {"expression": "1+1"}

    selected_step = _step_result(
        success=False,
        output={"result": {"answer": 2}, "tool_name": "calculator", "ok": False},
        error="tool failed",
    )
    assert (
        json_module._resolve_failure_error(select_output={"error": "bad model"}, selected_step=None)
        == "bad model"
    )
    assert (
        json_module._resolve_failure_error(
            select_output={},
            selected_step=selected_step,
        )
        == "tool failed"
    )
    assert (
        json_module._resolve_failure_error(
            select_output={},
            selected_step=_step_result(success=False, output={}, error=None),
        )
        == "Tool selection or execution failed."
    )

    assert json_module._resolve_failed_tool_output(None) == {}
    assert json_module._resolve_failed_tool_output(selected_step) == {"answer": 2}
    assert json_module._resolve_tool_results_for_failure(None) == []
    assert (
        json_module._resolve_tool_results_for_failure(
            _step_result(success=False, output="bad-output"),  # type: ignore[arg-type]
        )
        == []
    )
    assert (
        json_module._resolve_tool_results_for_failure(
            _step_result(success=False, output={"tool_name": None}),  # type: ignore[dict-item]
        )
        == []
    )
    tool_results = json_module._resolve_tool_results_for_failure(selected_step)
    assert len(tool_results) == 1
    assert tool_results[0].tool_name == "calculator"
    assert json_module._tool_error_payload("boom") == "boom"
    assert json_module._tool_error_payload({"type": "x", "message": "y"}) == {
        "type": "x",
        "message": "y",
    }
    assert json_module._tool_error_payload(object()) is None
    assert json_module._tool_step_id("calc-tool@v1") == "invoke_calc_tool_v1"
