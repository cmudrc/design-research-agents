"""Canonical delegate invocation helpers shared by workflow and implementations."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.workflow import (
    WorkflowDelegate,
    WorkflowDelegateRunner,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowObjectDelegate,
)


@dataclass(slots=True, frozen=True)
class DelegateInvocation:
    """Normalized delegate invocation payload."""

    result: ExecutionResult
    """Delegate execution result."""
    delegate_type: str
    """Resolved delegate category label (for example ``agent`` or ``workflow``)."""


def invoke_delegate(
    *,
    delegate: WorkflowDelegate,
    prompt: str,
    step_context: Mapping[str, object] | None,
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> DelegateInvocation:
    """Invoke a delegate and normalize its result/type metadata."""
    normalized_context = dict(step_context or {})
    if _is_workflow_object_delegate(delegate):
        workflow_result = _invoke_workflow_object_delegate(
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

    run_callable = getattr(delegate, "run", None)
    if not callable(run_callable):
        raise TypeError("Agent delegate must expose a callable run(prompt, ...) method.")
    agent_result = run_callable(
        prompt,
        request_id=request_id,
        dependencies=dependencies,
    )
    if not isinstance(agent_result, ExecutionResult):
        raise TypeError("Agent delegate must return ExecutionResult.")
    return DelegateInvocation(result=agent_result, delegate_type="agent")


def _invoke_workflow_object_delegate(
    *,
    delegate: WorkflowObjectDelegate,
    prompt: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> ExecutionResult:
    """Invoke a raw ``Workflow``-like delegate with input-mode adaptation."""
    input_schema = getattr(delegate, "_input_schema", None)
    if input_schema is None:
        workflow_input: str | Mapping[str, object] | None = prompt
    elif isinstance(input_schema, Mapping):
        workflow_input = {"prompt": prompt, "delegate_context": dict(step_context)}
    else:
        raise TypeError(
            "Workflow delegate must expose internal input schema via _input_schema "
            "as either a mapping or None."
        )
    workflow_result = delegate.run(
        workflow_input,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        request_id=request_id,
        dependencies=dependencies,
    )
    if not isinstance(workflow_result, ExecutionResult):
        raise TypeError("Workflow delegate must return ExecutionResult.")
    return workflow_result


def _is_workflow_object_delegate(delegate: WorkflowDelegate) -> TypeGuard[WorkflowObjectDelegate]:
    """Return whether delegate is a raw workflow object."""
    if not hasattr(delegate, "_input_schema"):
        return False
    input_schema = cast(Any, delegate)._input_schema
    if input_schema is not None and not isinstance(input_schema, Mapping):
        return False
    run_callable = getattr(delegate, "run", None)
    return callable(run_callable)


def _is_workflow_delegate_runner(
    delegate: WorkflowDelegate,
) -> TypeGuard[WorkflowDelegateRunner]:
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
    return not (
        first_parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and first_parameter.name in {"prompt", "input_data"}
    )


__all__ = ["DelegateInvocation", "invoke_delegate"]
