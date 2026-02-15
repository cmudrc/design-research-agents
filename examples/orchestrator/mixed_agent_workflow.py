"""Runnable example for the reusable mixed workflow orchestration chunk."""

import json

import design_research_agents as dra


def main() -> None:
    """Run the configured mixed workflow twice to demonstrate reusable routing."""
    llm_client = dra.llm.create_default_llm_client()
    tool_runtime = dra.tools.UnifiedToolRuntime()
    writer_agent = dra.agents.SingleStepDirectLLMAgent(llm_client=llm_client)

    orchestrator_steps = [
        dra.workflows.LogicStep(
            step_id="router",
            handler=lambda context: {
                "route": (
                    "template_path"
                    if str(context["prompt"]).lower().startswith("template:")
                    else "agent_path"
                )
            },
            route_map={
                "agent_path": ("draft_agent",),
                "template_path": ("draft_template",),
            },
        ),
        dra.workflows.AgentStep(
            step_id="draft_agent",
            agent_name="writer_agent",
            dependencies=("router",),
            prompt_builder=lambda context: (
                "Write one JSON object with keys title and summary for this request: "
                f"{context['prompt']}"
            ),
        ),
        dra.workflows.ToolStep(
            step_id="parse_agent_json",
            tool_name="text.extract_json",
            dependencies=("draft_agent",),
            input_builder=lambda context: {
                "text": context["dependency_results"]["draft_agent"]["output"]["output"][
                    "model_text"
                ]
            },
        ),
        dra.workflows.LogicStep(
            step_id="finalize_agent",
            dependencies=("parse_agent_json",),
            handler=lambda context: {
                "branch": "agent",
                "title": context["dependency_results"]["parse_agent_json"]["output"]["result"][
                    "json"
                ].get("title", ""),
                "summary": context["dependency_results"]["parse_agent_json"]["output"]["result"][
                    "json"
                ].get("summary", ""),
            },
        ),
        dra.workflows.LogicStep(
            step_id="draft_template",
            dependencies=("router",),
            handler=lambda context: {
                "title": "Template fallback brief",
                "summary": f"Template mode output for: {context['prompt']}",
            },
        ),
        dra.workflows.LogicStep(
            step_id="finalize_template",
            dependencies=("draft_template",),
            handler=lambda context: {
                "branch": "template",
                "title": context["dependency_results"]["draft_template"]["output"]["title"],
                "summary": context["dependency_results"]["draft_template"]["output"]["summary"],
            },
        ),
    ]

    orchestrator = dra.workflows.MixedAgentWorkflowOrchestrator(
        tool_runtime=tool_runtime,
        agents={"writer_agent": writer_agent},
        steps=orchestrator_steps,
    )

    agent_result = orchestrator.run(
        "Write a research brief for synthesis findings on prototype onboarding friction.",
        request_id="example-mixed-workflow-agent-branch",
    )
    template_result = orchestrator.run(
        "template: Draft a deterministic fallback brief for accessibility review findings.",
        request_id="example-mixed-workflow-template-branch",
    )

    print(
        json.dumps(
            {
                "agent_branch_run": agent_result.asdict(),
                "template_branch_run": template_result.asdict(),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
