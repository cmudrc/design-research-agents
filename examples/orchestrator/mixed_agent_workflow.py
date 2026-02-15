"""Mixed workflow runtime example combining logic, agent, and tool steps."""

from __future__ import annotations

import json
from collections.abc import Sequence

import design_research_agents as dra


def _build_writer_agent() -> dra.agents.SingleStepDirectLLMAgent:
    llm_client = dra.llm.create_default_llm_client()
    return dra.agents.SingleStepDirectLLMAgent(llm_client=llm_client)


def _build_steps() -> Sequence[
    dra.workflows.LogicStep | dra.workflows.AgentStep | dra.workflows.ToolStep
]:
    return [
        dra.workflows.LogicStep(
            step_id="router",
            handler=lambda context: {
                "route": "agent_path",
                "topic": context.get("topic", "workflow runtime"),
            },
            route_map={"agent_path": ("draft",), "other_path": ("skip_me",)},
        ),
        dra.workflows.AgentStep(
            step_id="draft",
            agent_name="writer_agent",
            dependencies=("router",),
            prompt_builder=lambda context: (
                "Write one JSON object proposal with title, summary, and priority about: "
                f"{context['dependency_results']['router']['output']['topic']}"
            ),
        ),
        dra.workflows.LogicStep(
            step_id="skip_me",
            dependencies=("router",),
            handler=lambda context: {"value": "This branch should not run."},
        ),
        dra.workflows.ToolStep(
            step_id="parse_json",
            tool_name="text.extract_json",
            dependencies=("draft",),
            input_builder=lambda context: {
                "text": context["dependency_results"]["draft"]["output"]["output"]["model_text"]
            },
        ),
        dra.workflows.ToolStep(
            step_id="persist_report",
            tool_name="fs.write_text",
            dependencies=("parse_json",),
            input_builder=lambda context: {
                "path": "artifacts/examples/mixed_workflow_report.json",
                "content": json.dumps(
                    {
                        "topic": "agent orchestration",
                        "draft": context["dependency_results"]["parse_json"]["output"]["result"][
                            "json"
                        ],
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                "overwrite": True,
            },
        ),
        dra.workflows.ToolStep(
            step_id="report_hash",
            tool_name="fs.hash",
            dependencies=("persist_report",),
            input_builder=lambda context: {
                "path": context["dependency_results"]["persist_report"]["output"]["result"]["path"],
                "algo": "sha256",
            },
        ),
        dra.workflows.LogicStep(
            step_id="finalize",
            dependencies=("parse_json", "report_hash"),
            handler=lambda context: {
                "title": context["dependency_results"]["parse_json"]["output"]["result"]["json"][
                    "title"
                ],
                "priority": context["dependency_results"]["parse_json"]["output"]["result"]["json"][
                    "priority"
                ],
                "report_digest": context["dependency_results"]["report_hash"]["output"]["result"][
                    "digest"
                ],
            },
        ),
    ]


def main() -> None:
    """Run the mixed workflow example and print the aggregated result."""
    runtime = dra.workflows.WorkflowRuntime(
        tool_runtime=dra.tools.UnifiedToolRuntime(),
        agents={"writer_agent": _build_writer_agent()},
    )
    result = runtime.run(_build_steps(), context={"topic": "agent orchestration"})
    print(result)


if __name__ == "__main__":
    main()
