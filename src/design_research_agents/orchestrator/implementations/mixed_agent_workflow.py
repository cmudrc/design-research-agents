"""Reusable mixed workflow orchestration chunk (logic + agent + tools)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents.agent import SingleStepDirectLLMAgent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.orchestrator import (
    AgentStep,
    LogicStep,
    ToolStep,
    WorkflowDelegate,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
    WorkflowOrchestrator,
    WorkflowResult,
)
from design_research_agents.contracts.tools import ToolRuntime

from .workflow_runtime import WorkflowRuntime

DEFAULT_WORKFLOW_TOPIC = "agent orchestration"
DEFAULT_REPORT_TOPIC = "agent orchestration"
DEFAULT_REPORT_PATH = "artifacts/examples/mixed_agent_workflow_report.json"
DEFAULT_WRITER_AGENT_NAME = "writer_agent"


def _dependency_output(
    context: Mapping[str, object],
    dependency_step_id: str,
) -> Mapping[str, object]:
    """Return one dependency ``output`` payload from workflow step context."""
    raw_dependency_results = context.get("dependency_results")
    if not isinstance(raw_dependency_results, Mapping):
        raise KeyError("Workflow context is missing 'dependency_results'.")
    raw_dependency_entry = raw_dependency_results.get(dependency_step_id)
    if not isinstance(raw_dependency_entry, Mapping):
        raise KeyError(f"Dependency '{dependency_step_id}' is unavailable in workflow context.")
    raw_output = raw_dependency_entry.get("output")
    if not isinstance(raw_output, Mapping):
        raise KeyError(f"Dependency '{dependency_step_id}' did not provide an 'output' payload.")
    return raw_output


def _dependency_result(
    context: Mapping[str, object],
    dependency_step_id: str,
) -> Mapping[str, object]:
    """Return one dependency ``output.result`` payload from workflow context."""
    raw_result = _dependency_output(context, dependency_step_id).get("result")
    if not isinstance(raw_result, Mapping):
        raise KeyError(f"Dependency '{dependency_step_id}' output did not include a result object.")
    return raw_result


def _dependency_agent_output(
    context: Mapping[str, object],
    dependency_step_id: str,
) -> Mapping[str, object]:
    """Return one dependency ``output.output`` payload for agent-step dependencies."""
    raw_agent_output = _dependency_output(context, dependency_step_id).get("output")
    if not isinstance(raw_agent_output, Mapping):
        raise KeyError(
            f"Dependency '{dependency_step_id}' output did not include an agent output object."
        )
    return raw_agent_output


def _dependency_result_json(
    context: Mapping[str, object],
    dependency_step_id: str,
) -> Mapping[str, object]:
    """Return one dependency ``output.result.json`` payload from workflow context."""
    raw_json = _dependency_result(context, dependency_step_id).get("json")
    if not isinstance(raw_json, Mapping):
        raise KeyError(f"Dependency '{dependency_step_id}' result did not include a json object.")
    return raw_json


def build_mixed_agent_workflow_steps(
    *,
    writer_agent_name: str = DEFAULT_WRITER_AGENT_NAME,
    report_topic: str = DEFAULT_REPORT_TOPIC,
    report_path: str = DEFAULT_REPORT_PATH,
) -> Sequence[LogicStep | AgentStep | ToolStep]:
    """Build workflow steps for a mixed logic/agent/tool orchestration."""
    return [
        LogicStep(
            step_id="router",
            handler=lambda context: {
                "route": "agent_path",
                "topic": context.get("topic", "workflow runtime"),
            },
            route_map={"agent_path": ("draft",), "other_path": ("skip_me",)},
        ),
        AgentStep(
            step_id="draft",
            agent_name=writer_agent_name,
            dependencies=("router",),
            prompt_builder=lambda context: (
                "Write one JSON object proposal with title, summary, and priority about: "
                f"{_dependency_output(context, 'router')['topic']}"
            ),
        ),
        LogicStep(
            step_id="skip_me",
            dependencies=("router",),
            handler=lambda context: {"value": "This branch should not run."},
        ),
        ToolStep(
            step_id="parse_json",
            tool_name="text.extract_json",
            dependencies=("draft",),
            input_builder=lambda context: {
                "text": _dependency_agent_output(context, "draft")["model_text"]
            },
        ),
        ToolStep(
            step_id="persist_report",
            tool_name="fs.write_text",
            dependencies=("parse_json",),
            input_builder=lambda context: {
                "path": report_path,
                "content": json.dumps(
                    {
                        "topic": report_topic,
                        "draft": _dependency_result_json(context, "parse_json"),
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                "overwrite": True,
            },
        ),
        ToolStep(
            step_id="report_hash",
            tool_name="fs.hash",
            dependencies=("persist_report",),
            input_builder=lambda context: {
                "path": _dependency_result(context, "persist_report")["path"],
                "algo": "sha256",
            },
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("parse_json", "report_hash"),
            handler=lambda context: {
                "title": _dependency_result_json(context, "parse_json")["title"],
                "priority": _dependency_result_json(context, "parse_json")["priority"],
                "report_digest": _dependency_result(context, "report_hash")["digest"],
            },
        ),
    ]


class MixedAgentWorkflowOrchestrator:
    """Configured mixed workflow orchestrator with fixed step topology."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        writer_agent: WorkflowDelegate | None = None,
        writer_agent_name: str = DEFAULT_WRITER_AGENT_NAME,
        topic: str = DEFAULT_WORKFLOW_TOPIC,
        report_topic: str = DEFAULT_REPORT_TOPIC,
        report_path: str = DEFAULT_REPORT_PATH,
    ) -> None:
        """Store dependencies and freeze mixed workflow steps and base context."""
        resolved_writer_agent = writer_agent or SingleStepDirectLLMAgent(llm_client=llm_client)
        self._runtime = WorkflowRuntime(
            tool_runtime=tool_runtime,
            agents={writer_agent_name: resolved_writer_agent},
        )
        self._default_context: dict[str, object] = {"topic": topic}
        self._steps = build_mixed_agent_workflow_steps(
            writer_agent_name=writer_agent_name,
            report_topic=report_topic,
            report_path=report_path,
        )

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "dag",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> WorkflowResult:
        """Execute the configured mixed workflow orchestration."""
        merged_context = dict(self._default_context)
        if context is not None:
            merged_context.update(context)
        return self._runtime.run(
            self._steps,
            context=merged_context,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            request_id=request_id,
            dependencies=dependencies,
        )


def mixed_agent_workflow(
    *,
    llm_client: LLMClient,
    tool_runtime: ToolRuntime,
    writer_agent: WorkflowDelegate | None = None,
    writer_agent_name: str = DEFAULT_WRITER_AGENT_NAME,
    topic: str = DEFAULT_WORKFLOW_TOPIC,
    report_topic: str = DEFAULT_REPORT_TOPIC,
    report_path: str = DEFAULT_REPORT_PATH,
) -> WorkflowOrchestrator:
    """Return a configured mixed workflow orchestration chunk."""
    return MixedAgentWorkflowOrchestrator(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        writer_agent=writer_agent,
        writer_agent_name=writer_agent_name,
        topic=topic,
        report_topic=report_topic,
        report_path=report_path,
    )


__all__ = [
    "DEFAULT_REPORT_PATH",
    "DEFAULT_REPORT_TOPIC",
    "DEFAULT_WORKFLOW_TOPIC",
    "DEFAULT_WRITER_AGENT_NAME",
    "MixedAgentWorkflowOrchestrator",
    "build_mixed_agent_workflow_steps",
    "mixed_agent_workflow",
]
