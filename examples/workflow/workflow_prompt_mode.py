"""Runnable example for ``Workflow`` in prompt-input mode."""

import json

from design_research_agents import LlamaCppServerLLMClient, Toolbox, Workflow
from design_research_agents.agent import SingleStepDirectLLMAgent
from design_research_agents.contracts.workflow import AgentStep, LogicStep, ToolStep


def main() -> None:
    """Run the configured mixed workflow twice to demonstrate reusable routing."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    writer_agent = SingleStepDirectLLMAgent(llm_client=llm_client)

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
                "Write one JSON object with keys title and summary for this request: "
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
                "title": "Template fallback brief",
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
    )

    agent_result = workflow.run(
        "Write a research brief for synthesis findings on prototype onboarding friction.",
        request_id="example-mixed-workflow-agent-branch",
    )
    template_result = workflow.run(
        "template: Draft a deterministic fallback brief for accessibility review findings.",
        request_id="example-mixed-workflow-template-branch",
    )

    def _summarize_run(result: object) -> dict[str, object]:
        """Return compact summary for one workflow run.

        Args:
            result: Workflow execution result object.

        Returns:
            Compact summary payload.
        """
        if not hasattr(result, "success") or not hasattr(result, "output"):
            return {"success": False, "error": "unexpected result type"}
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
            "execution_order": result.execution_order,
            "final_output": compact_final_output,
            "error": output.get("error"),
        }

    print(
        json.dumps(
            {
                "agent_branch_run": _summarize_run(agent_result),
                "template_branch_run": _summarize_run(template_result),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
