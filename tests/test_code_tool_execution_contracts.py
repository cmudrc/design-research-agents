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
    def __init__(
        self,
        *,
        fail: bool = False,
        non_mapping: bool = False,
        result_payload: object | None = None,
    ) -> None:
        self.fail = fail
        self.non_mapping = non_mapping
        self.result_payload = result_payload

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
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        if self.fail:
            return ToolResult(tool_name=tool_name, ok=False, error="boom")
        if self.non_mapping:
            return ToolResult(tool_name=tool_name, ok=True, result=["bad"])
        if self.result_payload is not None:
            return ToolResult(tool_name=tool_name, ok=True, result=self.result_payload)
        try:
            a_value = int(input.get("a", 0))
            b_value = int(input.get("b", 0))
        except Exception:
            a_value = 0
            b_value = 0
        return ToolResult(
            tool_name=tool_name,
            ok=True,
            result={"value": a_value + b_value},
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> _Runtime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


class _Stringable:
    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


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
    input_payload: Mapping[str, object] | None = None,
    dependencies: Mapping[str, object] | None = None,
) -> tuple[code_exec.CodeExecutionOutcome, list[ToolResult]]:
    compiled_code = compile_sandboxed_code(code_text)
    tool_results: list[ToolResult] = []
    output = execute_compiled_code(
        compiled_code=compiled_code,
        prompt="prompt",
        input_payload=dict(input_payload or {}),
        request_id="req",
        dependencies=dict(dependencies or {}),
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

    with pytest.raises(ValueError, match="Dunder attribute access is not allowed"):
        compile_sandboxed_code("value = (1).__class__")

    with pytest.raises(ValueError, match="Calling dunder names is not allowed"):
        compile_sandboxed_code("__safe__()")


def test_internal_payload_helpers_cover_bounds_and_proxy_mutations() -> None:
    proxy = code_exec._FinalOutputProxy()
    proxy["a"] = 1
    del proxy["a"]
    proxy.setdefault("b", 2)
    assert proxy.pop("b") == 2
    proxy.update({"c": 3})
    assert proxy.popitem() == ("c", 3)
    proxy.clear()
    assert proxy.was_mutated is True

    bounds = code_exec._PayloadBounds(max_depth=1, max_items=1, max_string_chars=4)
    normalized_mapping, mapping_changed = code_exec._normalize_json_value(
        {"alpha": {"nested": True}, "beta": 2},
        bounds=bounds,
    )
    assert mapping_changed is True
    assert normalized_mapping["alpha"] == "[truncated: max depth reached]"
    assert normalized_mapping["__truncated_items__"] == 1

    normalized_sequence, sequence_changed = code_exec._normalize_json_value(
        (1, 2, 3),
        bounds=bounds,
    )
    assert sequence_changed is True
    assert normalized_sequence == ["[truncated: max depth reached]", "[truncated_items:2]"]

    long_text = "x" * 100_000
    mapping_clone, mapping_clamped = code_exec._bounded_json_clone(
        {"blob": long_text},
        bounds_profiles=(code_exec._PayloadBounds(max_depth=8, max_items=8, max_string_chars=200_000),),
        fallback_root="mapping",
    )
    assert mapping_clamped is True
    assert mapping_clone == {
        "_truncated": True,
        "_reason": "[truncated: serialized payload exceeded 65536 bytes]",
    }

    sequence_clone, sequence_clamped = code_exec._bounded_json_clone(
        [long_text],
        bounds_profiles=(code_exec._PayloadBounds(max_depth=8, max_items=8, max_string_chars=200_000),),
        fallback_root="sequence",
    )
    assert sequence_clamped is True
    assert sequence_clone == ["[truncated: serialized payload exceeded 65536 bytes]"]

    scalar_clone, scalar_clamped = code_exec._bounded_json_clone(
        long_text,
        bounds_profiles=(code_exec._PayloadBounds(max_depth=8, max_items=8, max_string_chars=200_000),),
        fallback_root="scalar",
    )
    assert scalar_clamped is True
    assert isinstance(scalar_clone, str)
    assert len(scalar_clone) < len(long_text)
    assert "[truncated:" in scalar_clone

    clamped_tool_result = code_exec._clamp_tool_result(
        ToolResult(
            tool_name="sum",
            ok=True,
            result={"value": 3},
            warnings=["w" * 100_000],
            metadata={"blob": long_text},
        )
    )
    assert clamped_tool_result.metadata["metadata_clamped"] is True
    assert clamped_tool_result.metadata["warnings_clamped"] is True


def test_execute_compiled_code_uses_last_tool_result_as_final_output_fallback() -> None:
    output, tool_results = _run_code('call_tool("sum", {"a": 2, "b": 3})', runtime=_Runtime())
    assert output.final_output == {"value": 5}
    assert output.final_answer_called is False
    assert output.used_tool_output_fallback is True
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

    with pytest.raises(ValueError, match=r"final_answer\(\) payload must be a dict/object"):
        _run_code('final_answer(["not-a-mapping"])', runtime=_Runtime())


def test_execute_compiled_code_supports_explicit_final_answer_and_non_terminal_final_output() -> None:
    terminal_output, terminal_tool_results = _run_code('final_answer({"done": True})', runtime=_Runtime())
    assert terminal_output.final_output == {"done": True}
    assert terminal_output.final_answer_called is True
    assert terminal_output.used_tool_output_fallback is False
    assert terminal_tool_results == []

    step_output, step_tool_results = _run_code(
        "\n".join(
            [
                'result = call_tool("sum", {"a": 1, "b": 2})',
                'final_output = {"value": result["value"]}',
            ]
        ),
        runtime=_Runtime(),
    )
    assert step_output.final_output == {"value": 3}
    assert step_output.final_answer_called is False
    assert step_output.used_tool_output_fallback is False
    assert len(step_tool_results) == 1

    with pytest.raises(ValueError, match="call at least one tool or use `final_answer"):
        _run_code("final_output = {}", runtime=_Runtime())


def test_execute_compiled_code_exposes_json_safe_sandbox_inputs() -> None:
    output, _ = _run_code(
        'final_answer({"dep": dependencies["helper"], "count": input_payload["outer"]["count"]})',
        runtime=_Runtime(),
        input_payload={"outer": {"count": 3}},
        dependencies={"helper": _Stringable("helper-sentinel")},
    )
    assert output.final_answer_called is True
    assert output.final_output == {"dep": "helper-sentinel", "count": 3}


def test_execute_compiled_code_schema_validation_is_enforced() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _run_code('call_tool("sum", {"a": "x", "b": 2})', runtime=_Runtime())

    output, _ = _run_code(
        'call_tool("sum", {"a": "x", "b": 2})\nfinal_output = {"ok": True}',
        runtime=_Runtime(),
        validate_schema=False,
    )
    assert output.final_output == {"ok": True}


def test_execute_compiled_code_clamps_large_outputs() -> None:
    large_text = "x" * 100_000

    final_output, _ = _run_code(
        'final_answer({"blob": "x" * 100000})',
        runtime=_Runtime(),
    )
    assert isinstance(final_output.final_output["blob"], str)
    assert len(str(final_output.final_output["blob"])) < len(large_text)
    assert str(final_output.final_output["blob"]).endswith("...[truncated:100000]")

    fallback_output, tool_results = _run_code(
        'call_tool("sum", {"a": 1, "b": 2})',
        runtime=_Runtime(result_payload={"blob": large_text}),
    )
    assert isinstance(fallback_output.final_output["blob"], str)
    assert "[truncated:" in str(fallback_output.final_output["blob"])
    assert tool_results[-1].metadata["result_clamped"] is True


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
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        code_exec,
        "emit_guardrail_decision",
        lambda **kwargs: captured.append(dict(kwargs)),
    )

    monkeypatch.delattr(code_exec.signal, "SIGALRM", raising=False)
    with code_exec.execution_timeout(seconds=1):
        pass
    assert captured[-1]["guardrail"] == "execution_timeout_enforced"
    assert captured[-1]["decision"] == "warn"
    assert captured[-1]["details"] == {"seconds": 1, "enforced": False, "fallback": "unsupported_platform"}

    monkeypatch.setattr(code_exec.signal, "SIGALRM", 14, raising=False)
    monkeypatch.setattr(
        code_exec.signal,
        "signal",
        lambda *_args: (_ for _ in ()).throw(ValueError()),
    )
    with code_exec.execution_timeout(seconds=1):
        pass
    assert captured[-1]["guardrail"] == "execution_timeout_enforced"
    assert captured[-1]["decision"] == "warn"
    assert captured[-1]["details"] == {"seconds": 1, "enforced": False, "fallback": "not_main_thread"}


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
