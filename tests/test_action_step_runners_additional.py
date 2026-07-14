from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMChatParams, LLMMessage, LLMResponse
from design_research_agents._contracts._tools import ToolError, ToolResult, ToolRuntime, ToolSpec
from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._implementations._shared._agent_internal._code_action_step_runner import (
    CodeActionStepRunner,
)
from design_research_agents._implementations._shared._agent_internal._json_action_step_runner import (
    JsonActionStepRunner,
    _build_tool_input_builder,
    _invalid_selection_handler,
    _mapping_or_empty,
    _resolve_failed_tool_output,
    _resolve_failure_error,
    _resolve_tool_results_for_failure,
    _tool_error_payload,
    _tool_result_from_step_output,
    _tool_step_id,
)
from tests.helpers.workflow_stubs import SequenceLLMClient


class _CaptureChatClient:
    def __init__(self) -> None:
        self.params: LLMChatParams | None = None

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        del messages
        self.params = params
        return LLMResponse(text="final_output = {}", model=model, provider="capture")


class _ActionRuntime(ToolRuntime):
    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="sum",
                description="Add two numbers",
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            ),
            ToolSpec(
                name="fail",
                description="Always fail",
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
        if tool_name == "sum":
            return ToolResult(
                tool_name="sum",
                ok=True,
                result={
                    "value": int(input.get("a", 0)) + int(input.get("b", 0)),
                },
            )
        if tool_name == "fail":
            return ToolResult(
                tool_name="fail",
                ok=False,
                result={"detail": "failed"},
                error="boom",
            )
        return ToolResult(tool_name=tool_name, ok=False, result={}, error="unknown")

    def close(self) -> None:
        return None

    def __enter__(self) -> _ActionRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


class _EmptyRuntime(ToolRuntime):
    def list_tools(self) -> Sequence[ToolSpec]:
        return ()

    def invoke(
        self,
        tool_name: str,
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del tool_name, input, request_id, dependencies
        return ToolResult(tool_name="none", ok=False, result={}, error="no tools")

    def close(self) -> None:
        return None

    def __enter__(self) -> _EmptyRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


class _CalculatorRuntime(ToolRuntime):
    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="calculator",
                description="Evaluate arithmetic expressions",
                input_schema={
                    "type": "object",
                    "required": ["expression"],
                    "properties": {"expression": {"type": "string"}},
                    "additionalProperties": False,
                },
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
        if tool_name != "calculator":
            return ToolResult(tool_name=tool_name, ok=False, result={}, error="unknown")
        expression = str(input.get("expression", ""))
        try:
            result = float(eval(expression, {"__builtins__": {}}, {}))
        except Exception as exc:  # pragma: no cover - defensive safety for malformed expressions.
            return ToolResult(tool_name="calculator", ok=False, result={}, error=str(exc))
        return ToolResult(
            tool_name="calculator",
            ok=True,
            result={"expression": expression, "result": result},
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> _CalculatorRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


class _ReservedNameRuntime(ToolRuntime):
    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="final_answer",
                description="reserved",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
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
        del tool_name, input, request_id, dependencies
        return ToolResult(tool_name="final_answer", ok=True, result={})

    def close(self) -> None:
        return None

    def __enter__(self) -> _ReservedNameRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


def test_json_action_step_runner_success_path() -> None:
    runner = JsonActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=['{"tool_name":"sum","tool_input":{"a":2,"b":3}}']),
        tool_runtime=_ActionRuntime(),
    )

    result = runner.run("Compute 2 + 3")

    assert result.success is True
    assert result.output["tool_name"] == "sum"
    assert result.output["tool_output"] == {"value": 5}
    assert result.output["final_output"] == {"value": 5}
    assert result.output["action_type"] == "tool_call"
    assert result.output["final_answer_called"] is False
    assert len(result.tool_results) == 1
    assert result.tool_results[0].tool_name == "sum"
    assert result.output["workflow"]["success"] is True


def test_json_action_step_runner_supports_final_answer_action() -> None:
    runner = JsonActionStepRunner(
        llm_client=SequenceLLMClient(
            response_texts=['{"tool_name":"final_answer","tool_input":{"answer":"5"},"reason":"done"}']
        ),
        tool_runtime=_ActionRuntime(),
    )

    result = runner.run("Return the answer directly")

    assert result.success is True
    assert result.tool_results == []
    assert result.output["tool_name"] == "final_answer"
    assert result.output["final_output"] == {"answer": "5"}
    assert result.output["action_type"] == "final_answer"
    assert result.output["final_answer_called"] is True
    assert result.output["reason"] == "done"


def test_json_action_step_runner_rejects_legacy_tool_selection_payload_shape() -> None:
    runner = JsonActionStepRunner(
        llm_client=SequenceLLMClient(
            response_texts=['{"action":"TOOL_CALL","tool_names":["sum"],"reason":"legacy payload"}']
        ),
        tool_runtime=_ActionRuntime(),
    )

    result = runner.run("Compute 1 + 2")

    assert result.success is False
    assert result.output["tool_name"] is None
    assert "tool selection was invalid" in str(result.output["error"]).lower()
    assert result.metadata["stage"] == "tool_selection"


def test_json_action_step_runner_rejects_controller_style_payload_without_retry() -> None:
    runner = JsonActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=['{"continue": true, "thought": "Need to pick one tool."}']),
        tool_runtime=_ActionRuntime(),
    )

    result = runner.run("Compute 3 + 4")

    assert result.success is False
    assert "tool selection was invalid" in str(result.output["error"]).lower()


def test_json_action_step_runner_rejects_invalid_final_answer_payload() -> None:
    runner = JsonActionStepRunner(
        llm_client=SequenceLLMClient(
            response_texts=['{"tool_name":"final_answer","tool_input":"not-an-object","reason":"done"}']
        ),
        tool_runtime=_ActionRuntime(),
    )

    result = runner.run("Return the answer directly")

    assert result.success is False
    assert "final_answer" in str(result.output["error"])


def test_json_action_step_runner_invalid_selection_and_tool_failure() -> None:
    invalid_runner = JsonActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=['{"tool_name":"unknown","tool_input":{}}']),
        tool_runtime=_ActionRuntime(),
    )
    invalid_result = invalid_runner.run("Pick unknown tool")
    assert invalid_result.success is False
    assert invalid_result.output["tool_name"] is None
    assert "invalid" in str(invalid_result.output["error"]).lower()
    assert invalid_result.metadata["stage"] == "tool_selection"

    failing_runner = JsonActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=['{"tool_name":"fail","tool_input":{}}']),
        tool_runtime=_ActionRuntime(),
    )
    failing_result = failing_runner.run("Use failing tool")
    assert failing_result.success is False
    assert failing_result.output["tool_name"] == "fail"
    assert failing_result.output["tool_output"] == {"detail": "failed"}
    assert len(failing_result.tool_results) == 1
    assert failing_result.tool_results[0].ok is False


def test_json_action_step_runner_internal_helpers_cover_branch_paths() -> None:
    with pytest.raises(ValueError, match="Missing dependency_results"):
        _invalid_selection_handler({})

    with pytest.raises(ValueError, match="Missing select_tool result"):
        _invalid_selection_handler({"dependency_results": {}})

    with pytest.raises(ValueError, match="Invalid select_tool output"):
        _invalid_selection_handler({"dependency_results": {"select_tool": {"output": "bad"}}})

    with pytest.raises(ValueError, match="invalid model tool selection"):
        _invalid_selection_handler(
            {
                "dependency_results": {
                    "select_tool": {
                        "output": {
                            "error": " ",
                        }
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="chosen route is invalid"):
        _invalid_selection_handler(
            {
                "dependency_results": {
                    "select_tool": {
                        "output": {
                            "error": "chosen route is invalid",
                        }
                    }
                }
            }
        )

    builder = _build_tool_input_builder(expected_tool_name="sum")
    assert builder({}) == {}
    assert builder(
        {
            "dependency_results": {
                "select_tool": {
                    "output": {
                        "tool_name": "sum",
                        "tool_input": {"a": 1},
                    }
                }
            }
        }
    ) == {"a": 1}
    assert (
        builder(
            {
                "dependency_results": {
                    "select_tool": {
                        "output": {
                            "tool_name": "fail",
                            "tool_input": {"a": 1},
                        }
                    }
                }
            }
        )
        == {}
    )

    step_output = {
        "tool_name": "sum",
        "ok": True,
        "result": {"value": 1},
        "artifacts": [{"path": "artifact.txt", "mime": "text/plain"}],
        "warnings": ["warn"],
        "error": {"type": "ToolError", "message": "ignored"},
        "metadata": {"source": "test"},
    }
    rebuilt = _tool_result_from_step_output(step_output=step_output, fallback_tool_name="fallback")
    assert rebuilt.tool_name == "sum"
    assert rebuilt.result == {"value": 1}
    assert rebuilt.metadata == {"source": "test"}

    selected_step = WorkflowStepResult(
        step_id="invoke_fail",
        status="failed",
        success=False,
        output={"tool_name": "fail", "result": {"detail": "failed"}},
        error="tool failed",
    )
    assert (
        _resolve_failure_error(
            select_output={"error": "explicit"},
            selected_step=selected_step,
        )
        == "explicit"
    )
    assert _resolve_failure_error(select_output={}, selected_step=selected_step) == "tool failed"
    assert _resolve_failure_error(select_output={}, selected_step=None) == ("Tool selection or execution failed.")

    assert _resolve_failed_tool_output(None) == {}
    assert _resolve_failed_tool_output(selected_step) == {"detail": "failed"}
    assert _resolve_tool_results_for_failure(None) == []
    assert len(_resolve_tool_results_for_failure(selected_step)) == 1
    assert (
        _resolve_tool_results_for_failure(
            WorkflowStepResult(
                step_id="invoke_unknown",
                status="failed",
                success=False,
                output={"tool_name": 123},
                error="bad",
            )
        )
        == []
    )

    structured_error = ToolError(type="unit", message="boom")
    assert _tool_error_payload(structured_error) is structured_error
    assert _tool_error_payload({"type": "x", "message": "y"}) == {"type": "x", "message": "y"}
    assert _tool_error_payload("boom") == "boom"
    assert _tool_error_payload(5) is None

    assert _mapping_or_empty({"x": 1}) == {"x": 1}
    assert _mapping_or_empty("x") == {}
    assert _tool_step_id("a.b-c") == "invoke_a_b_c"
    assert _tool_step_id("___") == "invoke_tool"


def test_code_action_step_runner_keeps_trace_labels_out_of_provider_options() -> None:
    client = _CaptureChatClient()
    runner = CodeActionStepRunner(llm_client=client, tool_runtime=_ActionRuntime())

    response = runner._generate_code(
        prompt="Return an empty result.",
        allowed_tools=(),
        model="gemini-test",
        alternatives_prompt_target="user",
    )

    assert response.text == "final_output = {}"
    assert client.params is not None
    assert client.params.provider_options == {}


def test_code_action_step_runner_success_and_failure_modes() -> None:
    success_runner = CodeActionStepRunner(
        llm_client=SequenceLLMClient(
            response_texts=[
                "\n".join(
                    [
                        'calc = call_tool("sum", {"a": 2, "b": 3})',
                        'final_output = {"value": calc["value"]}',
                    ]
                )
            ]
        ),
        tool_runtime=_ActionRuntime(),
    )
    success = success_runner.run("Compute 2 + 3")
    assert success.success is True
    assert success.output["final_output"] == {"value": 5}
    assert success.output["final_answer_called"] is False
    assert success.output["action_type"] == "tool_call"
    assert success.output["tool_name"] == "sum"

    terminal_runner = CodeActionStepRunner(
        llm_client=SequenceLLMClient(
            response_texts=[
                "\n".join(
                    [
                        'calc = call_tool("sum", {"a": 4, "b": 5})',
                        'final_answer({"value": calc["value"]})',
                    ]
                )
            ]
        ),
        tool_runtime=_ActionRuntime(),
    )
    terminal_result = terminal_runner.run("Compute 4 + 5")
    assert terminal_result.success is True
    assert terminal_result.output["final_output"] == {"value": 9}
    assert terminal_result.output["final_answer_called"] is True
    assert terminal_result.output["action_type"] == "final_answer"

    validation_runner = CodeActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=["import os\nfinal_output = {}"]),
        tool_runtime=_ActionRuntime(),
    )
    validation_failure = validation_runner.run("Generate invalid code")
    assert validation_failure.success is False
    assert "validation" in str(validation_failure.output["error"]).lower()
    assert validation_failure.metadata["stage"] == "code_validation"

    execution_runner = CodeActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=['call_tool("fail", {})\nfinal_output = {"ok": False}']),
        tool_runtime=_ActionRuntime(),
    )
    execution_failure = execution_runner.run("Call failing tool")
    assert execution_failure.success is False
    assert "execution failed" in str(execution_failure.output["error"]).lower()
    assert execution_failure.metadata["stage"] == "code_execution"

    recovery_runner = CodeActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=['readme = call_tool("fs.read_text", {"path":"README.md"})']),
        tool_runtime=_CalculatorRuntime(),
        default_tools=({"tool_name": "calculator"},),
    )
    recovery_result = recovery_runner.run("Compute two design metrics: (12 * (4 + 1)) and (60 / 3).")
    assert recovery_result.success is True
    assert recovery_result.output["final_output"]["result"] == 20.0
    assert len(recovery_result.tool_results) == 2
    assert all(tool_result.tool_name == "calculator" for tool_result in recovery_result.tool_results)

    no_tools_runner = CodeActionStepRunner(
        llm_client=SequenceLLMClient(response_texts=[]),
        tool_runtime=_EmptyRuntime(),
    )
    no_tools_result = no_tools_runner.run("No tools")
    assert no_tools_result.success is False
    assert "no allowed tools" in str(no_tools_result.output["error"]).lower()
    assert no_tools_result.metadata["stage"] == "input_validation"

    with pytest.raises(ValueError, match="reserved tool name"):
        JsonActionStepRunner(
            llm_client=SequenceLLMClient(response_texts=['{"tool_name":"final_answer","tool_input":{}}']),
            tool_runtime=_ReservedNameRuntime(),
        )

    with pytest.raises(ValueError, match="reserved tool name"):
        CodeActionStepRunner(
            llm_client=SequenceLLMClient(response_texts=['final_answer({"done": True})']),
            tool_runtime=_ReservedNameRuntime(),
        )


class _FinalizePayloadAgent:
    def __init__(self, *, success: bool, payload: Mapping[str, object]) -> None:
        self._result = ExecutionResult(success=success, output=dict(payload))

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return self._result
