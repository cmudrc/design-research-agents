"""Run traced ``ReflexionPattern`` for iterative design-summary refinement.

Expected observations:
- ``critique_iterations`` increments during revision loops.
- ``approved`` reflects final critique acceptance.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import LlamaCppServerLLMClient, ReflexionPattern, Toolbox
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Run propose/critique refinement orchestration with tracing."""
    request_id = "example-workflow-propose-critic-design-001"
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = ReflexionPattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            tracer=make_tracer(),
        )
        result = workflow.run(
            prompt=(
                "Write and iteratively improve a short engineering design rationale for using "
                "modular connectors in field-serviceable devices."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/propose_critic.py",
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "approved": output.get("approved"),
        "critique_iterations": output.get("critique_iterations"),
        "proposal": output.get("proposal"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
