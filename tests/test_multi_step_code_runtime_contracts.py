from __future__ import annotations

from collections.abc import Sequence

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._memory import MemoryRecord, MemoryWriteRecord
from design_research_agents._contracts._termination import SOURCE_INVALID_PAYLOAD
from design_research_agents._contracts._tools import ToolResult, ToolSpec
from design_research_agents._implementations._shared._agent_internal import (
    _multi_step_code_runtime_helpers as code_runtime,
)

pytestmark = pytest.mark.contract


class _MemoryStore:
    def __init__(self, *, raise_error: bool = False) -> None:
        self.raise_error = raise_error
        self.records: list[MemoryWriteRecord] = []

    def write(
        self,
        records: Sequence[MemoryWriteRecord],
        *,
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        if self.raise_error:
            raise RuntimeError("memory write failed")
        self.records.extend(records)
        return [
            MemoryRecord(item_id=f"id-{index}", namespace=namespace, content=record.content)
            for index, record in enumerate(records, start=1)
        ]


def test_multi_step_code_runtime_helpers_cover_terminal_and_success_states() -> None:
    defaults = code_runtime.MultiStepCodeRunConfig(
        max_steps=3,
        max_tool_calls_per_step=2,
        execution_timeout_seconds=4,
        validate_tool_input_schema=True,
        stop_on_step_failure=True,
    )
    resolved = code_runtime.resolve_run_config(
        input_payload={
            "max_steps": 5,
            "max_tool_calls_per_step": 4,
            "execution_timeout_seconds": 6,
            "validate_tool_input_schema": False,
            "stop_on_step_failure": False,
        },
        defaults=defaults,
    )
    assert resolved.max_steps == 5
    assert resolved.stop_on_step_failure is False

    tool_specs = {
        "calculator": ToolSpec(
            name="calculator",
            description="calc",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
        "text.word_count": ToolSpec(
            name="text.word_count",
            description="count",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
    }
    all_tools_text = code_runtime.build_step_tools_text(
        tool_specs=tool_specs,
        default_tools_per_step=None,
    )
    assert "calculator" in all_tools_text
    selected_tools_text = code_runtime.build_step_tools_text(
        tool_specs=tool_specs,
        default_tools_per_step=[
            {"tool_name": "calculator"},
            {"tool_name": "missing"},
            {"name": ""},
        ],
    )
    assert "calculator" in selected_tools_text
    assert "text.word_count" not in selected_tools_text

    assert code_runtime.summarize_code_action(generated_code=123) == ""
    assert code_runtime.summarize_observation(final_output={"x": 1}, error=None).startswith("{")
    assert code_runtime.summarize_observation(final_output={}, error=" boom ") == "error: boom"

    success_step = ExecutionResult(
        success=True,
        output={"final_output": {"value": 9}},
        tool_results=[],
        model_response=None,
    )
    completion = code_runtime.resolve_step_completion(
        step_result=success_step,
        step_outputs=[],
        memory=[],
        final_output={},
        stop_on_step_failure=True,
    )
    assert completion[0] == {"value": 9}
    assert completion[2] is True

    no_tool_call_step = ExecutionResult(
        success=False,
        output={"error": "Generated code must call at least one tool."},
        tool_results=[],
        model_response=None,
    )
    step_outputs = [{"step": 1}]
    memory = [{"kind": "task"}, {"kind": "observation"}]
    completion_no_tool = code_runtime.resolve_step_completion(
        step_result=no_tool_call_step,
        step_outputs=step_outputs,
        memory=memory,
        final_output={"value": 1},
        stop_on_step_failure=True,
    )
    assert completion_no_tool[1] == "continuation_stopped:empty_step"
    assert completion_no_tool[2] is False

    failed_step = ExecutionResult(
        success=False,
        output={"error": "other failure"},
        tool_results=[],
        model_response=None,
    )
    completion_fail = code_runtime.resolve_step_completion(
        step_result=failed_step,
        step_outputs=[],
        memory=[],
        final_output={},
        stop_on_step_failure=True,
    )
    assert completion_fail[3] == "other failure"

    final_success = code_runtime.build_code_final_result(
        final_state={
            "memory": [],
            "continuation_trace": [],
            "retrieval_trace": [],
            "memory_errors": [],
            "step_outputs": [{"success": True}],
            "tool_results": [],
            "final_output": {"done": True},
            "terminated_reason": "ok",
            "last_model_response": LLMResponse(text="x"),
            "fatal_error": None,
            "fatal_metadata": {},
        },
        request_id="req",
        dependencies={},
        max_steps=3,
        max_tool_calls_per_step=2,
        execution_timeout_seconds=4,
        validate_tool_input_schema=True,
        normalize_generated_code_per_step=False,
        stop_on_step_failure=True,
        alternatives_prompt_target="user",
        continuation_memory_tail_items=6,
        step_memory_tail_items=4,
        memory_namespace="default",
        memory_read_top_k=3,
        memory_write_observations=True,
        default_tools_per_step=None,
        memory_store_enabled=True,
    )
    assert final_success.success is True

    final_failure = code_runtime.build_code_final_result(
        final_state={
            "memory": [],
            "continuation_trace": [],
            "retrieval_trace": [],
            "memory_errors": [],
            "step_outputs": [{"success": False}],
            "tool_results": [ToolResult(tool_name="x", ok=False, error="boom")],
            "final_output": {},
            "terminated_reason": "failed",
            "last_model_response": None,
            "fatal_error": "fatal",
            "fatal_metadata": {"stage": "x"},
        },
        request_id="req",
        dependencies={},
        max_steps=3,
        max_tool_calls_per_step=2,
        execution_timeout_seconds=4,
        validate_tool_input_schema=True,
        normalize_generated_code_per_step=False,
        stop_on_step_failure=True,
        alternatives_prompt_target="user",
        continuation_memory_tail_items=6,
        step_memory_tail_items=4,
        memory_namespace="default",
        memory_read_top_k=3,
        memory_write_observations=True,
        default_tools_per_step=None,
        memory_store_enabled=True,
    )
    assert final_failure.success is False
    assert final_failure.output["error"] == "fatal"

    store = _MemoryStore()
    write_error = code_runtime.write_step_observation(
        memory_store=store,
        namespace="notes",
        task_prompt="task",
        step_number=1,
        continuation_reason="reason",
        step_success=True,
        generated_code='call_tool("x", {})',
        final_output={"x": 1},
        error=None,
    )
    assert write_error is None
    assert store.records

    failing_store = _MemoryStore(raise_error=True)
    write_failure = code_runtime.write_step_observation(
        memory_store=failing_store,
        namespace="notes",
        task_prompt="task",
        step_number=1,
        continuation_reason="reason",
        step_success=False,
        generated_code="",
        final_output={},
        error="boom",
    )
    assert "memory write failed" in str(write_failure)

    retrieval_trace: list[dict[str, object]] = []
    memory_errors: list[str] = []
    code_runtime.append_retrieval_trace(
        retrieval_trace=retrieval_trace,
        memory_errors=memory_errors,
        retrieval_error="read failed",
        step_number=2,
        namespace="notes",
        retrieved_match_count=0,
    )
    assert memory_errors == ["read(step 2): read failed"]
    assert retrieval_trace[0]["step"] == 2

    step_input = code_runtime.build_step_input(
        normalized_input={"prompt": "base"},
        step_prompt="step",
        max_tool_calls_per_step=3,
        execution_timeout_seconds=5,
        validate_tool_input_schema=True,
        alternatives_prompt_target="user",
    )
    assert step_input["prompt"] == "step"

    stop_state = code_runtime.build_continuation_stop_state(
        memory=[],
        continuation_trace=[],
        retrieval_trace=[],
        memory_errors=[],
        step_outputs=[],
        tool_results=[],
        final_output={},
        last_model_response=None,
        continue_source=SOURCE_INVALID_PAYLOAD,
    )
    assert stop_state["should_continue"] is False
    assert stop_state["fatal_error"] is not None

    assert code_runtime.is_no_tool_call_step_failure(error="Generated code must call at least one tool.")
