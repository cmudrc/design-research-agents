"""Mixed workflow runtime example combining logic, agent, and tool steps."""

from __future__ import annotations

from collections.abc import Sequence

from _orchestrator_example_support import SequenceResponseLLMClient

import design_research_agents


def _build_writer_agent() -> design_research_agents.DirectLLMAgent:
    llm_client = SequenceResponseLLMClient(
        response_texts=[
            "Draft proposal: use deterministic workflows with agent delegation.",
        ]
    )
    return design_research_agents.DirectLLMAgent(llm_client=llm_client)


def _build_steps() -> Sequence[
    design_research_agents.LogicStep
    | design_research_agents.AgentStep
    | design_research_agents.ToolStep
]:
    return [
        design_research_agents.LogicStep(
            step_id="router",
            handler=lambda context: {
                "route": "agent_path",
                "topic": context.get("topic", "workflow runtime"),
            },
            route_map={"agent_path": ("draft",), "other_path": ("skip_me",)},
        ),
        design_research_agents.AgentStep(
            step_id="draft",
            agent_name="writer_agent",
            dependencies=("router",),
            prompt_builder=lambda context: (
                "Write one concise proposal sentence about: "
                f"{context['dependency_results']['router']['output']['topic']}"
            ),
        ),
        design_research_agents.LogicStep(
            step_id="skip_me",
            dependencies=("router",),
            handler=lambda context: {"value": "This branch should not run."},
        ),
        design_research_agents.ToolStep(
            step_id="stats",
            tool_name="text_stats_tool",
            dependencies=("draft",),
            input_builder=lambda context: {
                "text": context["dependency_results"]["draft"]["output"]["output"]["model_text"]
            },
        ),
        design_research_agents.LogicStep(
            step_id="finalize",
            dependencies=("stats",),
            handler=lambda context: {
                "draft_word_count": context["dependency_results"]["stats"]["output"]["result"][
                    "word_count"
                ]
            },
        ),
    ]


def main() -> None:
    runtime = design_research_agents.WorkflowRuntime(
        tool_runtime=design_research_agents.BaseToolRuntime(),
        agents={"writer_agent": _build_writer_agent()},
    )
    result = runtime.run(_build_steps(), context={"topic": "agent orchestration"})
    print(result)


if __name__ == "__main__":
    main()
