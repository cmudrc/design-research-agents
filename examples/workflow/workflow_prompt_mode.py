"""Run traced ``Workflow(input_mode="prompt")`` for design-brief routing.

Expected observations:
- agent and template branches both produce finalized title/summary payloads.
- ``execution_order`` shows branch-specific step execution.
- trace metadata is present for each run.
"""

from __future__ import annotations

from design_research_agents import (
    AgentStep,
    DirectLLMCall,
    LlamaCppServerLLMClient,
    LogicStep,
    Toolbox,
    ToolStep,
    Workflow,
)
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def _summarize_run(result: object, request_id: str) -> dict[str, object]:
    if not hasattr(result, "success") or not hasattr(result, "output"):
        return {
            "success": False,
            "error": "unexpected result type",
            "trace": trace_info(request_id),
        }
    output = result.output if isinstance(result.output, dict) else {}
    final_output = output.get("final_output")
    if isinstance(final_output, dict):
        compact_final_output = {
            "branch": final_output.get("branch"),
            "title": final_output.get("title"),
            "summary": final_output.get("summary"),
        }
    else:
        compact_final_output = final_output
    return {
        "success": result.success,
        "execution_order": list(result.execution_order),
        "final_output": compact_final_output,
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }


def main() -> None:
    """Run reusable prompt-mode workflow for two routed design requests."""
    tracer = make_tracer()
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    writer_agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)

    workflow_steps = [
        LogicStep(
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
        AgentStep(
            step_id="draft_agent",
            delegate=writer_agent,
            dependencies=("router",),
            prompt_builder=lambda context: (
                "Write one JSON object with keys title and summary for this design request: "
                f"{context['prompt']}"
            ),
        ),
        ToolStep(
            step_id="parse_agent_json",
            tool_name="text.extract_json",
            dependencies=("draft_agent",),
            input_builder=lambda context: {
                "text": context["dependency_results"]["draft_agent"]["output"]["output"][
                    "model_text"
                ]
            },
        ),
        LogicStep(
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
        LogicStep(
            step_id="draft_template",
            dependencies=("router",),
            handler=lambda context: {
                "title": "Template fallback design brief",
                "summary": f"Template mode output for: {context['prompt']}",
            },
        ),
        LogicStep(
            step_id="finalize_template",
            dependencies=("draft_template",),
            handler=lambda context: {
                "branch": "template",
                "title": context["dependency_results"]["draft_template"]["output"]["title"],
                "summary": context["dependency_results"]["draft_template"]["output"]["summary"],
            },
        ),
    ]

    workflow = Workflow(
        tool_runtime=tool_runtime,
        steps=workflow_steps,
        input_mode="prompt",
        tracer=tracer,
    )

    agent_request_id = "example-workflow-prompt-design-agent-001"
    template_request_id = "example-workflow-prompt-design-template-001"
    try:
        agent_result = workflow.run(
            "Draft a design brief for reducing onboarding friction in a medical-device setup flow.",
            request_id=agent_request_id,
        )
        template_result = workflow.run(
            (
                "template: Produce a deterministic fallback brief for "
                "manufacturability review findings."
            ),
            request_id=template_request_id,
        )
    finally:
        llm_client.close()

    print_json(
        {
            "agent_branch_run": _summarize_run(agent_result, agent_request_id),
            "template_branch_run": _summarize_run(template_result, template_request_id),
        }
    )


if __name__ == "__main__":
    main()
