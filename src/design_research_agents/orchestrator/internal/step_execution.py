"""Workflow step executors for tool, agent, and logic step kinds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.orchestrator import (
    AgentStep,
    LogicStep,
    ToolStep,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowStepResult,
)
from design_research_agents.contracts.tools import ToolRuntime

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
    """Execute one tool step and return normalized workflow step result."""
    if tool_runtime is None:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
            error="Tool step requires a configured tool_runtime.",
            metadata={"stage": "tool_binding", "tool_name": step.tool_name},
        )

    available_tools = {tool_spec.name for tool_spec in tool_runtime.list_tools()}
    if step.tool_name not in available_tools:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
            error=f"Unknown tool '{step.tool_name}'.",
            metadata={"stage": "tool_binding", "tool_name": step.tool_name},
        )

    try:
        tool_input = resolve_tool_input(step=step, step_context=step_context)
    except Exception as exc:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
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
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
            error=str(exc),
            metadata={"stage": "execution", "tool_name": step.tool_name},
        )

    serialized_output = asdict(tool_result)
    if not tool_result.ok:
        if tool_result.error is not None:
            tool_error_message = tool_result.error.message
        else:
            tool_error_message = "Tool invocation failed."
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
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
    agents: Mapping[str, Agent],
    step: AgentStep,
    step_id: str,
    step_context: Mapping[str, object],
    request_id: str,
    execution_mode: WorkflowExecutionMode,
    failure_policy: WorkflowFailurePolicy,
    dependencies: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one agent step and return normalized workflow step result."""
    selected_agent = agents.get(step.agent_name)
    if selected_agent is None:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
            error=f"Unknown agent '{step.agent_name}'.",
            metadata={"stage": "agent_binding", "agent_name": step.agent_name},
        )

    try:
        prompt = resolve_agent_prompt(step=step, step_context=step_context)
    except Exception as exc:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
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

    try:
        agent_result = selected_agent.run(
            prompt,
            request_id=f"{request_id}:workflow:{step_id}",
            dependencies=invocation_dependencies,
        )
    except Exception as exc:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
            error=str(exc),
            metadata={"stage": "execution", "agent_name": step.agent_name},
        )

    serialized_output = agent_result.asdict()
    if not agent_result.success:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output=serialized_output,
            error=str(agent_result.output.get("error", "Agent execution failed.")),
            metadata={"stage": "execution", "agent_name": step.agent_name},
        )

    return WorkflowStepResult(
        step_id=step_id,
        status="completed",
        success=True,
        output=serialized_output,
        metadata={"stage": "execution", "agent_name": step.agent_name},
    )


def run_logic_step(
    *,
    step: LogicStep,
    step_id: str,
    step_context: Mapping[str, object],
) -> WorkflowStepResult:
    """Execute one logic step handler and normalize result payload."""
    try:
        step_output = dict(step.handler(step_context))
    except Exception as exc:
        return WorkflowStepResult(
            step_id=step_id,
            status="failed",
            success=False,
            output={},
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
