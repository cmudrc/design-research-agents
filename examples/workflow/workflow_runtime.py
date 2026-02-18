"""Runnable entrypoint demonstrating the public ``Workflow`` facade."""

from design_research_agents.contracts.workflow import LogicStep
from design_research_agents.workflow import Workflow


def main() -> None:
    """Run a minimal logic-only workflow and print serialized output."""
    workflow = Workflow(
        tool_runtime=None,
        input_mode="schema",
        steps=[
            LogicStep(
                step_id="hello_workflow",
                handler=lambda _context: {"message": "workflow runtime ready"},
            )
        ],
    )
    result = workflow.run({}, execution_mode="sequential")
    print(result.asdict())


if __name__ == "__main__":
    main()
