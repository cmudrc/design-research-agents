"""Canonical delegate invocation helpers shared by workflow and implementations."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

from design_research_agents._contracts._delegate import Delegate
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._workflow import (
    DelegateRunner,
    DelegateTarget,
    WorkflowDelegate,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class DelegateInvocation:
    """Normalized delegate invocation payload."""

    result: ExecutionResult
    """Delegate execution result."""
    delegate_type: str
    """Resolved delegate category label (for example ``delegate`` or ``workflow``)."""


def invoke_delegate(
    *,
    delegate: DelegateTarget,
    prompt: str,
    step_context: Mapping[str, object] | None,
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> DelegateInvocation:
    """Invoke a delegate and normalize its result/type metadata."""
    normalized_context = dict(step_context or {})
    # Workflow-object delegates need input-shape adaptation before invocation.
    if _is_workflow_delegate(delegate):
        workflow_result = _invoke_workflow_delegate(
            delegate=delegate,
            prompt=prompt,
            step_context=normalized_context,
            request_id=request_id,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            dependencies=dependencies,
        )
        return DelegateInvocation(result=workflow_result, delegate_type="workflow")

    if _is_workflow_delegate_runner(delegate):
        # Runner delegates accept context-first signatures, so inject prompt into context explicitly.
        nested_context = dict(normalized_context)
        nested_context["prompt"] = prompt
        workflow_result = delegate.run(
            context=nested_context,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            request_id=request_id,
            dependencies=dependencies,
        )
        if not isinstance(workflow_result, ExecutionResult):
            raise TypeError("Workflow delegate must return ExecutionResult.")
        return DelegateInvocation(result=workflow_result, delegate_type="workflow")

    compile_callable = getattr(delegate, "compile", None)
    delegate_compile_stub = getattr(Delegate, "compile", None)
    compile_impl = getattr(type(delegate), "compile", None)
    if callable(compile_callable) and compile_impl is not delegate_compile_stub:
        compiled_execution = compile_callable(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        compiled_run = getattr(compiled_execution, "run", None)
        if not callable(compiled_run):
            raise TypeError("Delegate compile() must return an object with a callable run() method.")
        delegate_result = compiled_run()
    else:
        run_callable = getattr(delegate, "run", None)
        if not callable(run_callable):
            raise TypeError("Delegate must expose a callable compile(prompt, ...) or run(prompt, ...) method.")
        delegate_result = run_callable(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
    if not isinstance(delegate_result, ExecutionResult):
        raise TypeError("Delegate execution must return ExecutionResult.")
    return DelegateInvocation(result=delegate_result, delegate_type="delegate")


def _invoke_workflow_delegate(
    *,
    delegate: WorkflowDelegate,
    prompt: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> ExecutionResult:
    """Invoke a raw ``Workflow``-like delegate with input-mode adaptation."""
    input_schema = getattr(delegate, "_input_schema", None)
    # Prompt-mode workflows consume plain strings; schema-mode workflows expect structured mapping input.
    if input_schema is None:
        workflow_input: str | Mapping[str, object] | None = prompt
    elif isinstance(input_schema, Mapping):
        workflow_input = {"prompt": prompt, "delegate_context": dict(step_context)}
    else:
        raise TypeError(
            "Workflow delegate must expose internal input schema via _input_schema as either a mapping or None."
        )
    workflow_result = delegate.run(
        input=workflow_input,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        request_id=request_id,
        dependencies=dependencies,
    )
    if not isinstance(workflow_result, ExecutionResult):
        raise TypeError("Workflow delegate must return ExecutionResult.")
    return workflow_result


def _is_workflow_delegate(delegate: DelegateTarget) -> TypeGuard[WorkflowDelegate]:
    """Return whether delegate is a raw workflow object."""
    if not hasattr(delegate, "_input_schema"):
        return False
    input_schema = cast(Any, delegate)._input_schema
    if input_schema is not None and not isinstance(input_schema, Mapping):
        return False
    run_callable = getattr(delegate, "run", None)
    return callable(run_callable)


def _is_workflow_delegate_runner(
    delegate: DelegateTarget,
) -> TypeGuard[DelegateRunner]:
    """Return whether delegate ``run`` signature matches workflow-runner style."""
    run_callable = getattr(delegate, "run", None)
    if run_callable is None:
        return False
    try:
        signature = inspect.signature(run_callable)
    except (TypeError, ValueError):
        return False

    parameters = list(signature.parameters.values())
    if not parameters:
        return True
    first_parameter = parameters[0]
    # Distinguish workflow-style run(context=...) from agent-style run(prompt, ...).
    return not (
        first_parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and first_parameter.name in {"prompt", "input"}
    )


__all__ = ["DelegateInvocation", "invoke_delegate"]
