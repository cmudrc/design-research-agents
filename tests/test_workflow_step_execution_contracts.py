from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.memory import (
    MemoryRecord,
    MemorySearchQuery,
    MemoryWriteRecord,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.contracts.workflow import (
    AgentStep,
    MemoryReadStep,
    MemoryWriteStep,
    ToolStep,
)
from design_research_agents.workflow.internal.step_execution import (
    run_agent_step,
    run_memory_read_step,
    run_memory_write_step,
    run_tool_step,
)

pytestmark = pytest.mark.contract


class _Runtime(ToolRuntime):
    def __init__(
        self,
        *,
        tool_names: Sequence[str] = ("sum",),
        raise_error: bool = False,
        result_ok: bool = True,
        include_error: bool = True,
    ) -> None:
        self._tool_names = tuple(tool_names)
        self._raise_error = raise_error
        self._result_ok = result_ok
        self._include_error = include_error

    def list_tools(self) -> Sequence[ToolSpec]:
        return tuple(
            ToolSpec(
                name=name,
                description=name,
                input_schema={"type": "object", "additionalProperties": True},
                output_schema={"type": "object", "additionalProperties": True},
            )
            for name in self._tool_names
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
        if self._raise_error:
            raise RuntimeError("tool exploded")
        if not self._result_ok:
            return ToolResult(
                tool_name=tool_name,
                ok=False,
                result={"input": dict(input_dict)},
                error="tool failed" if self._include_error else None,
            )
        return ToolResult(tool_name=tool_name, ok=True, result={"input": dict(input_dict)})


class _AgentSuccess:
    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del request_id, dependencies
        return ExecutionResult(success=True, output={"echo": prompt})


class _AgentFailure:
    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(success=False, output={"error": "agent failed"})


class _AgentRaises:
    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        raise RuntimeError("agent exploded")


class _WorkflowDelegateSuccess:
    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del execution_mode, failure_policy, request_id, dependencies
        return ExecutionResult(success=True, output={"prompt": (context or {}).get("prompt", "")})


class _WorkflowDelegateFailure:
    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del context, execution_mode, failure_policy, request_id, dependencies
        return ExecutionResult(success=False, output={"error": "nested failed"})


class _WorkflowDelegateRaises:
    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del context, execution_mode, failure_policy, request_id, dependencies
        raise RuntimeError("nested exploded")


class _MemoryStore:
    def __init__(self, *, raise_search: bool = False, raise_write: bool = False) -> None:
        self.raise_search = raise_search
        self.raise_write = raise_write

    def search(self, query: MemorySearchQuery) -> list[MemoryRecord]:
        if self.raise_search:
            raise RuntimeError("search exploded")
        return [
            MemoryRecord(
                item_id="id-1",
                namespace=query.namespace,
                content=f"match:{query.text}",
                metadata={"top_k": query.top_k},
            )
        ]

    def write(
        self,
        records: Sequence[MemoryWriteRecord],
        *,
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        if self.raise_write:
            raise RuntimeError("write exploded")
        return [
            MemoryRecord(item_id=f"id-{index}", namespace=namespace, content=record.content)
            for index, record in enumerate(records, start=1)
        ]


def _common_context() -> dict[str, object]:
    return {
        "prompt": "task",
        "dependency_results": {},
        "step_results": {},
    }


def test_run_tool_step_covers_failure_and_success_paths() -> None:
    step = ToolStep(step_id="tool", tool_name="sum", input_data={"a": 1})

    missing_runtime = run_tool_step(
        tool_runtime=None,
        step=step,
        step_id="tool",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert missing_runtime.success is False

    unknown_tool = run_tool_step(
        tool_runtime=_Runtime(tool_names=("other",)),
        step=step,
        step_id="tool",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert "Unknown tool" in str(unknown_tool.error)

    bad_input = run_tool_step(
        tool_runtime=_Runtime(),
        step=ToolStep(
            step_id="tool",
            tool_name="sum",
            input_builder=lambda _ctx: (_ for _ in ()).throw(ValueError("bad input")),
        ),
        step_id="tool",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert bad_input.metadata["stage"] == "input_build"

    invoke_error = run_tool_step(
        tool_runtime=_Runtime(raise_error=True),
        step=step,
        step_id="tool",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert invoke_error.metadata["stage"] == "execution"

    explicit_failure = run_tool_step(
        tool_runtime=_Runtime(result_ok=False, include_error=True),
        step=step,
        step_id="tool",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert explicit_failure.error == "tool failed"

    generic_failure = run_tool_step(
        tool_runtime=_Runtime(result_ok=False, include_error=False),
        step=step,
        step_id="tool",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert generic_failure.error == "Tool invocation failed."

    success = run_tool_step(
        tool_runtime=_Runtime(),
        step=step,
        step_id="tool",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert success.success is True
    assert success.status == "completed"


def test_run_agent_step_covers_agent_and_delegate_paths() -> None:
    bad_prompt = run_agent_step(
        step=AgentStep(
            step_id="agent",
            delegate=_AgentSuccess(),
            prompt_builder=lambda _ctx: (_ for _ in ()).throw(ValueError("bad prompt")),
        ),
        step_id="agent",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert bad_prompt.metadata["stage"] == "input_build"

    nested_raises = run_agent_step(
        step=AgentStep(step_id="agent", delegate=_WorkflowDelegateRaises(), prompt="hello"),
        step_id="agent",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert nested_raises.success is False

    nested_failure = run_agent_step(
        step=AgentStep(step_id="agent", delegate=_WorkflowDelegateFailure(), prompt="hello"),
        step_id="agent",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert nested_failure.error == "Nested workflow execution failed."

    nested_success = run_agent_step(
        step=AgentStep(step_id="agent", delegate=_WorkflowDelegateSuccess(), prompt="hello"),
        step_id="agent",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert nested_success.success is True

    agent_raises = run_agent_step(
        step=AgentStep(step_id="agent", delegate=_AgentRaises(), prompt="hello"),
        step_id="agent",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert agent_raises.success is False

    agent_failure = run_agent_step(
        step=AgentStep(step_id="agent", delegate=_AgentFailure(), prompt="hello"),
        step_id="agent",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert agent_failure.error == "agent failed"

    agent_success = run_agent_step(
        step=AgentStep(step_id="agent", delegate=_AgentSuccess(), prompt="hello"),
        step_id="agent",
        step_context=_common_context(),
        request_id="req",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={},
    )
    assert agent_success.success is True


def test_run_memory_read_step_covers_binding_input_execution_and_success() -> None:
    missing_store = run_memory_read_step(
        memory_store=None,
        step=MemoryReadStep(step_id="read", query_builder=lambda _ctx: "query"),
        step_id="read",
        step_context=_common_context(),
    )
    assert missing_store.success is False

    query_error = run_memory_read_step(
        memory_store=_MemoryStore(),
        step=MemoryReadStep(
            step_id="read",
            query_builder=lambda _ctx: (_ for _ in ()).throw(ValueError("bad query")),
        ),
        step_id="read",
        step_context=_common_context(),
    )
    assert query_error.metadata["stage"] == "input_build"

    bad_type = run_memory_read_step(
        memory_store=_MemoryStore(),
        step=MemoryReadStep(step_id="read", query_builder=lambda _ctx: object()),
        step_id="read",
        step_context=_common_context(),
    )
    assert "string or mapping" in str(bad_type.error)

    search_error = run_memory_read_step(
        memory_store=_MemoryStore(raise_search=True),
        step=MemoryReadStep(step_id="read", query_builder=lambda _ctx: "query"),
        step_id="read",
        step_context=_common_context(),
    )
    assert search_error.metadata["stage"] == "execution"

    success = run_memory_read_step(
        memory_store=_MemoryStore(),
        step=MemoryReadStep(
            step_id="read",
            query_builder=lambda _ctx: {
                "query": "design",
                "top_k": 0,
                "metadata_filters": {"k": 1},
            },
            namespace="notes",
            top_k=5,
        ),
        step_id="read",
        step_context=_common_context(),
    )
    assert success.success is True
    assert success.output["query"]["top_k"] == 1
    assert success.output["count"] == 1


def test_run_memory_write_step_covers_binding_input_execution_and_success() -> None:
    missing_store = run_memory_write_step(
        memory_store=None,
        step=MemoryWriteStep(step_id="write", records_builder=lambda _ctx: ["a"]),
        step_id="write",
        step_context=_common_context(),
    )
    assert missing_store.success is False

    build_error = run_memory_write_step(
        memory_store=_MemoryStore(),
        step=MemoryWriteStep(
            step_id="write",
            records_builder=lambda _ctx: (_ for _ in ()).throw(ValueError("bad records")),
        ),
        step_id="write",
        step_context=_common_context(),
    )
    assert build_error.metadata["stage"] == "input_build"

    wrong_shape = run_memory_write_step(
        memory_store=_MemoryStore(),
        step=MemoryWriteStep(step_id="write", records_builder=lambda _ctx: {"content": "x"}),
        step_id="write",
        step_context=_common_context(),
    )
    assert "sequence" in str(wrong_shape.error)

    missing_content = run_memory_write_step(
        memory_store=_MemoryStore(),
        step=MemoryWriteStep(step_id="write", records_builder=lambda _ctx: [{"item_id": "x"}]),
        step_id="write",
        step_context=_common_context(),
    )
    assert "must include 'content'" in str(missing_content.error)

    write_error = run_memory_write_step(
        memory_store=_MemoryStore(raise_write=True),
        step=MemoryWriteStep(step_id="write", records_builder=lambda _ctx: ["a"]),
        step_id="write",
        step_context=_common_context(),
    )
    assert write_error.metadata["stage"] == "execution"

    success = run_memory_write_step(
        memory_store=_MemoryStore(),
        step=MemoryWriteStep(
            step_id="write",
            records_builder=lambda _ctx: [
                "first",
                {"content": "second", "metadata": {"k": 1}, "item_id": " item-2 "},
                MemoryWriteRecord(content="third"),
            ],
            namespace="notes",
        ),
        step_id="write",
        step_context=_common_context(),
    )
    assert success.success is True
    assert success.output["written"] == 3
    assert success.output["namespace"] == "notes"
