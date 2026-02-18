"""Runnable entrypoint demonstrating the core ``WorkflowRuntime`` implementation."""

from design_research_agents.contracts.workflow import LogicStep
from design_research_agents.workflow.internal.workflow_runtime import WorkflowRuntime


def main() -> None:
    """Run a minimal logic-only workflow and print serialized output."""
    runtime = WorkflowRuntime()
    result = runtime.run(
        [
            LogicStep(
                step_id="hello_workflow",
                handler=lambda _context: {"message": "workflow runtime ready"},
            )
        ],
        execution_mode="sequential",
    )
    print(result.asdict())


if __name__ == "__main__":
    main()
