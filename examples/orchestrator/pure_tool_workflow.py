"""Pure workflow runtime example with chained tool and logic steps."""

from __future__ import annotations

import design_research_agents


def main() -> None:
    runtime = design_research_agents.WorkflowRuntime(
        tool_runtime=design_research_agents.BaseToolRuntime(),
    )
    steps = [
        design_research_agents.ToolStep(
            step_id="compute",
            tool_name="calculator_tool",
            input_data={"expression": "12 * (4 + 1)"},
        ),
        design_research_agents.LogicStep(
            step_id="format",
            dependencies=("compute",),
            handler=lambda context: {
                "summary_text": (
                    "The calculator result is "
                    f"{int(context['dependency_results']['compute']['output']['result']['result'])}."
                )
            },
        ),
        design_research_agents.ToolStep(
            step_id="stats",
            tool_name="text_stats_tool",
            dependencies=("format",),
            input_builder=lambda context: {
                "text": context["dependency_results"]["format"]["output"]["summary_text"]
            },
        ),
        design_research_agents.LogicStep(
            step_id="finalize",
            dependencies=("stats",),
            handler=lambda context: {
                "word_count": context["dependency_results"]["stats"]["output"]["result"][
                    "word_count"
                ],
                "line_count": context["dependency_results"]["stats"]["output"]["result"][
                    "line_count"
                ],
            },
        ),
    ]

    result = runtime.run(steps, execution_mode="sequential")
    print(result.asdict())


if __name__ == "__main__":
    main()
