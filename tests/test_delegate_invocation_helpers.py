from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._runtime._common import _delegate_invocation as delegate_impl


class _AgentDelegate:
    def __init__(self) -> None:
        self.prompt: str | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del request_id, dependencies
        self.prompt = prompt
        return ExecutionResult(success=True, output={"model_text": prompt})


class _RunnerDelegate:
    def __init__(self) -> None:
        self.context: Mapping[str, object] | None = None

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
        self.context = context
        return ExecutionResult(success=True, output={"ok": True})


class _WorkflowPromptObject:
    _input_schema = None

    def __init__(self) -> None:
        self.input: str | Mapping[str, object] | None = None

    def run(
        self,
        input: str | Mapping[str, object] | None = None,
        *,
        execution_mode: str = "sequential",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del execution_mode, failure_policy, request_id, dependencies
        self.input = input
        return ExecutionResult(success=True, output={"ok": True})


class _WorkflowSchemaObject:
    _input_schema: ClassVar[dict[str, object]] = {"type": "object"}

    def __init__(self) -> None:
        self.input: str | Mapping[str, object] | None = None

    def run(
        self,
        input: str | Mapping[str, object] | None = None,
        *,
        execution_mode: str = "sequential",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del execution_mode, failure_policy, request_id, dependencies
        self.input = input
        return ExecutionResult(success=True, output={"ok": True})


class _BadSchemaWorkflowObject:
    _input_schema = "bad-schema"

    def run(
        self,
        input: str | Mapping[str, object] | None = None,
        *,
        execution_mode: str = "sequential",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del input, execution_mode, failure_policy, request_id, dependencies
        return ExecutionResult(success=True, output={"ok": True})


class _BadResultWorkflowObject:
    _input_schema = None

    def run(
        self,
        input: str | Mapping[str, object] | None = None,
        *,
        execution_mode: str = "sequential",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> str:
        del input, execution_mode, failure_policy, request_id, dependencies
        return "bad"


class _BadResultRunner:
    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: str = "dag",
        failure_policy: str = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> str:
        del context, execution_mode, failure_policy, request_id, dependencies
        return "bad"


class _BadResultAgent:
    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> str:
        del prompt, request_id, dependencies
        return "bad"


class _NoRun:
    pass


def _invoke(delegate: object, *, step_context: Mapping[str, object] | None = None) -> object:
    return delegate_impl.invoke_delegate(
        delegate=delegate,  # type: ignore[arg-type]
        prompt="task prompt",
        step_context=step_context,
        request_id="req-1",
        execution_mode="sequential",
        failure_policy="skip_dependents",
        dependencies={"dep": True},
    )


def test_invoke_delegate_agent_path() -> None:
    delegate = _AgentDelegate()
    invocation = _invoke(delegate)
    assert isinstance(invocation, delegate_impl.DelegateInvocation)
    assert invocation.delegate_type == "agent"
    assert invocation.result.success is True
    assert delegate.prompt == "task prompt"


def test_invoke_delegate_workflow_runner_path() -> None:
    delegate = _RunnerDelegate()
    invocation = _invoke(delegate, step_context={"k": 1})
    assert invocation.delegate_type == "workflow"
    assert invocation.result.success is True
    assert isinstance(delegate.context, Mapping)
    assert delegate.context.get("k") == 1
    assert delegate.context.get("prompt") == "task prompt"


def test_invoke_delegate_workflow_object_prompt_mode() -> None:
    delegate = _WorkflowPromptObject()
    invocation = _invoke(delegate)
    assert invocation.delegate_type == "workflow"
    assert delegate.input == "task prompt"


def test_invoke_delegate_workflow_object_schema_mode() -> None:
    delegate = _WorkflowSchemaObject()
    invocation = _invoke(delegate, step_context={"k": 1})
    assert invocation.delegate_type == "workflow"
    assert isinstance(delegate.input, Mapping)
    assert delegate.input.get("prompt") == "task prompt"
    assert delegate.input.get("delegate_context") == {"k": 1}


def test_invoke_delegate_rejects_bad_workflow_input_schema() -> None:
    with pytest.raises(TypeError, match="input schema"):
        delegate_impl._invoke_workflow_object_delegate(
            delegate=_BadSchemaWorkflowObject(),  # type: ignore[arg-type]
            prompt="task prompt",
            step_context={},
            request_id="req-1",
            execution_mode="sequential",
            failure_policy="skip_dependents",
            dependencies={},
        )


def test_invoke_delegate_rejects_bad_result_types() -> None:
    with pytest.raises(TypeError, match="Workflow delegate must return ExecutionResult"):
        _invoke(_BadResultRunner())

    with pytest.raises(TypeError, match="Workflow delegate must return ExecutionResult"):
        _invoke(_BadResultWorkflowObject())

    with pytest.raises(TypeError, match="Agent delegate must return ExecutionResult"):
        _invoke(_BadResultAgent())


def test_invoke_delegate_rejects_missing_run() -> None:
    with pytest.raises(TypeError, match="callable run"):
        _invoke(_NoRun())


def test_delegate_type_guards_cover_runner_and_object_paths() -> None:
    assert delegate_impl._is_workflow_object_delegate(_WorkflowPromptObject()) is True
    assert delegate_impl._is_workflow_object_delegate(_BadSchemaWorkflowObject()) is False
    assert delegate_impl._is_workflow_object_delegate(_RunnerDelegate()) is False
    assert delegate_impl._is_workflow_delegate_runner(_RunnerDelegate()) is True
    assert delegate_impl._is_workflow_delegate_runner(_AgentDelegate()) is False
