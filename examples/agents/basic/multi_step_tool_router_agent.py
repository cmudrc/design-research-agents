"""Runnable example showing JSON-mode router special-case execution lifecycle."""

import json
from collections.abc import Mapping, Sequence

from design_research_agents import LlamaCppServerLLMClient
from design_research_agents.agent import MultiStepAgent
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec


class _RouterDemoToolRuntime(ToolRuntime):
    """Minimal arg-less runtime so JSON mode enters router-special-case."""

    def list_tools(self) -> Sequence[ToolSpec]:
        """Expose available demo tools.

        Returns:
            Tool specifications available to the runtime.
        """
        return (
            ToolSpec(
                name="calculator",
                description="Compute simple demo expressions.",
                input_schema={"type": "object", "additionalProperties": False},
                output_schema={"type": "object", "additionalProperties": True},
            ),
        )

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one demo tool.

        Args:
            tool_name: Tool name to invoke.
            input_dict: Tool input payload.
            request_id: Invocation request id.
            dependencies: Dependency mapping for invocation.

        Returns:
            Tool invocation result payload.
        """
        del request_id, dependencies
        expression = str(input_dict.get("expression", "12 * (4 + 1)"))
        if tool_name != "calculator":
            return ToolResult(tool_name=tool_name, ok=False, result={}, error="unknown tool")
        result = 60.0 if expression == "12 * (4 + 1)" else 0.0
        return ToolResult(
            tool_name="calculator",
            ok=True,
            result={"expression": expression, "result": result},
        )


def main() -> None:
    """Execute one multi-step tool-router run and print the resulting result."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = _RouterDemoToolRuntime()
    try:
        agent = MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
        )
        result = agent.run(
            prompt="Compute 12 * (4 + 1), then stop with a final structured output.",
            request_id="example-multi-step-tool-router-agent-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "steps_executed": output.get("steps_executed"),
        "tool_results_count": len(result.tool_results),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
