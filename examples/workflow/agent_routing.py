"""Runnable example for intent routing across concrete DRA agent delegates.

The pattern is workflow-native and built on ``WorkflowRuntime`` primitives.
"""

from design_research_agents import (
    DirectLLMCall,
    LlamaCppServerLLMClient,
    MultiStepAgent,
    RouterPattern,
    Toolbox,
)


def main() -> None:
    """Route one prompt to the best delegate agent and print the final result."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()

    direct_llm_agent = DirectLLMCall(llm_client=llm_client)
    json_tool_agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=1,
    )

    workflow = RouterPattern(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        alternatives={
            "direct_llm_agent": direct_llm_agent,
            "json_tool_agent": json_tool_agent,
        },
        alternative_descriptions={
            "direct_llm_agent": "Use for direct text-only responses with no runtime tools.",
            "json_tool_agent": "Use for requests that should invoke runtime tools.",
        },
    )

    # Internal routing is performed by MultiStepAgent(mode="json", max_steps=1)
    # in the arg-less-tools router special-case.
    result = workflow.run(
        prompt="Calculate this expression and return the numeric result: 12 * (4 + 1)",
        request_id="example-agent-routing-workflow-001",
    )
    print(result)


if __name__ == "__main__":
    main()
