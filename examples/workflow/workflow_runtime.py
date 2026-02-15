"""Runnable entrypoint demonstrating the core ``WorkflowRuntime`` implementation."""

import design_research_agents as dra


def main() -> None:
    """Run a minimal logic-only workflow and print serialized output."""
    runtime = dra.workflows.WorkflowRuntime()
    result = runtime.run(
        [
            dra.workflows.LogicStep(
                step_id="hello_workflow",
                handler=lambda _context: {"message": "workflow runtime ready"},
            )
        ],
        execution_mode="sequential",
    )
    print(result.asdict())


if __name__ == "__main__":
    main()
