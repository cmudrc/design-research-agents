"""Workflow step executors for tool, agent, and logic step kinds."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import asdict
from typing import TypeGuard, cast

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import (
    AgentStep,
    LogicStep,
    ToolStep,
    WorkflowDelegate,
    WorkflowDelegateRunner,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowStepResult,
)

from .step_context import build_invocation_dependencies, resolve_agent_prompt, resolve_tool_input


def run_tool_step(
    *,
    tool_runtime: ToolRuntime | None,
    step: ToolStep,
    step_id: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one tool step and return normalized workflow step result.

    Args:
        tool_runtime: Tool runtime used to resolve and invoke tools.
        step: Tool step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.
        request_id: Workflow request id for scoped tool invocation ids.
        execution_mode: Effective workflow execution mode.
        failure_policy: Effective workflow failure policy.
        dependencies: Run dependency mapping passed into tool invocation.

    Returns:
        Normalized workflow step result for this tool step.
    """
    if tool_runtime is None:
        return _failed_step_result(
            step_id=step_id,
            error="Tool step requires a configured tool_runtime.",
            metadata={"stage": "tool_binding", "tool_name": step.tool_name},
        )

    available_tools = {tool_spec.name for tool_spec in tool_runtime.list_tools()}
    if step.tool_name not in available_tools:
        return _failed_step_result(
            step_id=step_id,
            error=f"Unknown tool '{step.tool_name}'.",
            metadata={"stage": "tool_binding", "tool_name": step.tool_name},
        )

    try:
        tool_input = resolve_tool_input(step=step, step_context=step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "tool_name": step.tool_name},
        )

    invocation_dependencies = build_invocation_dependencies(
        base_dependencies=dependencies,
        step_id=step_id,
        request_id=request_id,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        step_context=step_context,
    )

    try:
        tool_result = tool_runtime.invoke(
            step.tool_name,
            tool_input,
            request_id=f"{request_id}:workflow:{step_id}",
            dependencies=invocation_dependencies,
        )
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "execution", "tool_name": step.tool_name},
        )

    serialized_output = asdict(tool_result)
    if not tool_result.ok:
        if tool_result.error is not None:
            tool_error_message = tool_result.error.message
        else:
            tool_error_message = "Tool invocation failed."
        return _failed_step_result(
            step_id=step_id,
            output=serialized_output,
            error=tool_error_message,
            metadata={"stage": "execution", "tool_name": step.tool_name},
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=serialized_output,
        metadata={"stage": "execution", "tool_name": step.tool_name},
    )


def run_agent_step(
    *,
    agents: Mapping[str, WorkflowDelegate],
    step: AgentStep,
    step_id: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one agent-like step and return normalized workflow step result.

    Args:
        agents: Delegate registry for ``AgentStep`` resolution.
        step: Agent step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.
        request_id: Workflow request id for scoped delegate invocation ids.
        execution_mode: Effective workflow execution mode.
        failure_policy: Effective workflow failure policy.
        dependencies: Run dependency mapping passed into delegate invocation.

    Returns:
        Normalized workflow step result for this agent step.
    """
    selected_delegate = agents.get(step.agent_name)
    if selected_delegate is None:
        return _failed_step_result(
            step_id=step_id,
            error=f"Unknown agent '{step.agent_name}'.",
            metadata={"stage": "agent_binding", "agent_name": step.agent_name},
        )

    try:
        prompt = resolve_agent_prompt(step=step, step_context=step_context)
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "input_build", "agent_name": step.agent_name},
        )

    invocation_dependencies = build_invocation_dependencies(
        base_dependencies=dependencies,
        step_id=step_id,
        request_id=request_id,
        execution_mode=execution_mode,
        failure_policy=failure_policy,
        step_context=step_context,
    )

    request_scope = f"{request_id}:workflow:{step_id}"

    if _is_workflow_delegate_runner(selected_delegate):
        nested_context = dict(step_context)
        nested_context["prompt"] = prompt
        try:
            workflow_result = selected_delegate.run(
                context=nested_context,
                execution_mode=execution_mode,
                failure_policy=failure_policy,
                request_id=request_scope,
                dependencies=invocation_dependencies,
            )
        except Exception as exc:
            return _failed_step_result(
                step_id=step_id,
                error=str(exc),
                metadata={
                    "stage": "execution",
                    "agent_name": step.agent_name,
                    "delegate_type": "workflow",
                },
            )

        serialized_output = workflow_result.asdict()
        if not workflow_result.success:
            return _failed_step_result(
                step_id=step_id,
                output=serialized_output,
                error="Nested workflow execution failed.",
                metadata={
                    "stage": "execution",
                    "agent_name": step.agent_name,
                    "delegate_type": "workflow",
                },
            )

        return WorkflowStepResult(
            step_id=step_id,
            status="completed",
            success=True,
            output=serialized_output,
            metadata={
                "stage": "execution",
                "agent_name": step.agent_name,
                "delegate_type": "workflow",
            },
        )

    selected_agent = cast(Agent, selected_delegate)
    try:
        agent_result = selected_agent.run(
            prompt,
            request_id=request_scope,
            dependencies=invocation_dependencies,
        )
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={
                "stage": "execution",
                "agent_name": step.agent_name,
                "delegate_type": "agent",
            },
        )

    serialized_output = agent_result.asdict()
    if not agent_result.success:
        return _failed_step_result(
            step_id=step_id,
            output=serialized_output,
            error=str(agent_result.output.get("error", "Agent execution failed.")),
            metadata={
                "stage": "execution",
                "agent_name": step.agent_name,
                "delegate_type": "agent",
            },
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=serialized_output,
        metadata={
            "stage": "execution",
            "agent_name": step.agent_name,
            "delegate_type": "agent",
        },
    )


def _is_workflow_delegate_runner(
    delegate: WorkflowDelegate,
) -> TypeGuard[WorkflowDelegateRunner]:
    """Return true when delegate ``run`` signature matches workflow style.

    Args:
        delegate: Delegate object registered for an ``AgentStep``.

    Returns:
        ``True`` when delegate ``run`` signature matches workflow-style execution.
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
        and first_parameter.name == "prompt"
    )


def run_logic_step(
    *,
    step: LogicStep,
    step_id: str,
    step_context: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one logic step handler and normalize result payload.

    Args:
        step: Logic step definition to execute.
        step_id: Step identifier for result metadata.
        step_context: Step execution context with dependency outputs.

    Returns:
        Normalized workflow step result for this logic step.
    """
    try:
        step_output = dict(step.handler(step_context))
    except Exception as exc:
        return _failed_step_result(
            step_id=step_id,
            error=str(exc),
            metadata={"stage": "execution"},
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=step_output,
        metadata={"stage": "execution"},
    )


def _failed_step_result(
    *,
    step_id: str,
    error: str,
    metadata: Mapping[str, object],
    output: Mapping[str, object] | None = None,
) -> WorkflowStepResult:
    """Build a standardized failed ``WorkflowStepResult`` payload.

    Args:
        step_id: Step identifier for the failed step.
        error: Human-readable failure message.
        metadata: Additional metadata to attach to the result.
        output: Optional partial output captured before failure.

    Returns:
        Failed workflow step result.
    """
    return WorkflowStepResult(
        step_id=step_id,
        status="failed",
        success=False,
        output=dict(output or {}),
        error=error,
        metadata=dict(metadata),
    )
