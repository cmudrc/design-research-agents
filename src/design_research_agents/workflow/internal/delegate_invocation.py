"""Shared delegate invocation helpers for workflow and pattern runtimes."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeGuard

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
    """Invoke a delegate and normalize its result/type metadata.

    Args:
        delegate: Delegate object to invoke.
        prompt: Prompt text for agent-like delegates.
        step_context: Optional workflow step context.
        request_id: Request id scoped for this delegate invocation.
        execution_mode: Effective workflow execution mode.
        failure_policy: Effective workflow failure policy.
        dependencies: Invocation dependency payload.

    Returns:
        Normalized delegate invocation payload.

    Raises:
        TypeError: Raised when delegate output is invalid.
    """
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
    """Invoke a raw ``Workflow``-like delegate with input-mode adaptation.

    Args:
        delegate: Workflow-like delegate object.
        prompt: Prompt text to adapt into workflow input.
        step_context: Optional step context mapping.
        request_id: Request id for this delegate invocation.
        execution_mode: Workflow execution mode.
        failure_policy: Workflow failure policy.
        dependencies: Dependency mapping passed through to workflow execution.

    Returns:
        Delegate execution result.

    Raises:
        TypeError: Raised when input mode is unsupported or result type is invalid.
    """
    input_mode = str(getattr(delegate, "_input_mode", "")).strip().lower()
    if input_mode == "prompt":
        workflow_input: str | Mapping[str, object] | None = prompt
    elif input_mode == "schema":
        workflow_input = {"prompt": prompt, "delegate_context": dict(step_context)}
    else:
        raise TypeError(
            "Workflow delegate must expose internal input mode via _input_mode "
            "as either 'prompt' or 'schema'."
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
    """Return whether delegate is a raw workflow object.

    Args:
        delegate: Delegate candidate.

    Returns:
        ``True`` when delegate has workflow input mode metadata and callable ``run``.
    """
    input_mode = getattr(delegate, "_input_mode", None)
    if not isinstance(input_mode, str):
        return False
    run_callable = getattr(delegate, "run", None)
    if not callable(run_callable):
        return False
    return input_mode.strip().lower() in {"prompt", "schema"}


def _is_workflow_delegate_runner(
    delegate: WorkflowDelegate,
) -> TypeGuard[WorkflowDelegateRunner]:
    """Return whether delegate ``run`` signature matches workflow-runner style.

    Args:
        delegate: Delegate candidate.

    Returns:
        ``True`` when the delegate expects workflow-style ``run(context=...)`` invocation.
    """
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
