from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents._implementations._shared._agent_internal import (
    _code_tool_agent_execution as code_exec,
)
from design_research_agents._implementations._shared._agent_internal._code_tool_agent_execution import (
    compile_sandboxed_code,
    execute_compiled_code,
    failure_result,
    validate_field_type,
    validate_input_against_schema,
)
from design_research_agents._implementations._shared._agent_internal._code_tool_agent_parsing import (
    AllowedTool,
)

pytestmark = pytest.mark.contract


class _Runtime(ToolRuntime):
    def __init__(self, *, fail: bool = False, non_mapping: bool = False) -> None:
        self.fail = fail
        self.non_mapping = non_mapping

    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="sum",
                description="sum",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
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
        if self.fail:
            return ToolResult(tool_name=tool_name, ok=False, error="boom")
        if self.non_mapping:
            return ToolResult(tool_name=tool_name, ok=True, result=["bad"])
        try:
            a_value = int(input_dict.get("a", 0))
            b_value = int(input_dict.get("b", 0))
        except Exception:
            a_value = 0
            b_value = 0
        return ToolResult(
            tool_name=tool_name,
            ok=True,
            result={"value": a_value + b_value},
        )


def _allowed_tools() -> tuple[AllowedTool, ...]:
    return (
        AllowedTool(
            tool_name="sum",
            description="sum",
            input_schema={
                "type": "object",
                "required": ["a", "b"],
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "additionalProperties": False,
            },
        ),
    )


def _run_code(
    code_text: str,
    *,
    runtime: ToolRuntime,
    max_tool_calls: int = 3,
    validate_schema: bool = True,
) -> tuple[dict[str, object], list[ToolResult]]:
    compiled_code = compile_sandboxed_code(code_text)
    tool_results: list[ToolResult] = []
    output = execute_compiled_code(
        compiled_code=compiled_code,
        prompt="prompt",
        input_payload={},
        request_id="req",
        dependencies={},
        allowed_tools=_allowed_tools(),
        tool_runtime=runtime,
        max_tool_calls=max_tool_calls,
        execution_timeout_seconds=5,
        validate_tool_input_schema=validate_schema,
        tool_results=tool_results,
    )
    return output, tool_results


def test_compile_sandboxed_code_rejects_empty_and_banned_syntax() -> None:
    with pytest.raises(ValueError, match="empty"):
        compile_sandboxed_code("")

    with pytest.raises(ValueError, match="Unsupported syntax node"):
        compile_sandboxed_code("import os")

    with pytest.raises(ValueError, match="banned name"):
        compile_sandboxed_code("value = open('x')")


def test_execute_compiled_code_uses_last_tool_result_as_final_output_fallback() -> None:
    output, tool_results = _run_code('call_tool("sum", {"a": 2, "b": 3})', runtime=_Runtime())
    assert output == {"value": 5}
    assert len(tool_results) == 1


def test_execute_compiled_code_rejects_bad_tool_input_and_tool_budget_overrun() -> None:
    with pytest.raises(ValueError, match="mapping/object"):
        _run_code('call_tool("sum", 3)\nfinal_output = {}', runtime=_Runtime())

    with pytest.raises(RuntimeError, match="Tool call limit exceeded"):
        _run_code(
            'call_tool("sum", {"a": 1, "b": 1})\ncall_tool("sum", {"a": 1, "b": 1})',
            runtime=_Runtime(),
            max_tool_calls=1,
        )


def test_execute_compiled_code_handles_tool_failures_and_result_shapes() -> None:
    with pytest.raises(RuntimeError, match="failed: boom"):
        _run_code('call_tool("sum", {"a": 1, "b": 2})', runtime=_Runtime(fail=True))

    with pytest.raises(RuntimeError, match="non-object payload"):
        _run_code('call_tool("sum", {"a": 1, "b": 2})', runtime=_Runtime(non_mapping=True))

    with pytest.raises(ValueError, match="dict/object"):
        _run_code(
            'call_tool("sum", {"a": 1, "b": 2})\nfinal_output = ["not-a-mapping"]',
            runtime=_Runtime(),
        )


def test_execute_compiled_code_schema_validation_is_enforced() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _run_code('call_tool("sum", {"a": "x", "b": 2})', runtime=_Runtime())

    output, _ = _run_code(
        'call_tool("sum", {"a": "x", "b": 2})\nfinal_output = {"ok": True}',
        runtime=_Runtime(),
        validate_schema=False,
    )
    assert output == {"ok": True}


def test_validation_helpers_cover_required_additional_and_field_types() -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {
            "a": {"type": "integer"},
            "name": {"type": "string"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "obj": {"type": "object"},
            "items": {"type": "array"},
        },
        "additionalProperties": False,
    }

    validate_input_against_schema(
        input_payload={
            "a": 1,
            "name": "n",
            "ratio": 1.2,
            "flag": True,
            "obj": {"k": 1},
            "items": [1, 2],
        },
        input_schema=schema,
    )

    with pytest.raises(ValueError, match="Missing required"):
        validate_input_against_schema(input_payload={}, input_schema=schema)

    with pytest.raises(ValueError, match="Unexpected tool input field"):
        validate_input_against_schema(input_payload={"a": 1, "extra": 2}, input_schema=schema)

    with pytest.raises(ValueError, match="must be an integer"):
        validate_field_type(field_name="a", field_value="x", field_schema={"type": "integer"})

    with pytest.raises(ValueError, match="must be a number"):
        validate_field_type(field_name="ratio", field_value=True, field_schema={"type": "number"})


def test_execution_timeout_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(code_exec.signal, "SIGALRM", raising=False)
    with code_exec.execution_timeout(seconds=1):
        pass

    monkeypatch.setattr(code_exec.signal, "SIGALRM", 14, raising=False)
    monkeypatch.setattr(
        code_exec.signal,
        "signal",
        lambda *_args: (_ for _ in ()).throw(ValueError()),
    )
    with code_exec.execution_timeout(seconds=1):
        pass


def test_failure_result_includes_generated_code_and_optional_raw_code() -> None:
    result = failure_result(
        error="boom",
        model_response=None,
        tool_results=[],
        request_id="req-1",
        dependencies={"upstream": 1},
        metadata={"stage": "unit"},
        generated_code="final_output = {}",
        raw_generated_code="```python\\nfinal_output = {}\\n```",
    )

    assert result.success is False
    assert result.output["generated_code"] == "final_output = {}"
    assert "raw_generated_code" in result.output
    assert result.metadata["request_id"] == "req-1"
