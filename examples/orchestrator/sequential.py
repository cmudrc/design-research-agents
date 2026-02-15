"""Runnable example for ``SequentialOrchestrator``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import design_research_agents
from design_research_agents.contracts.orchestrator import WorkflowNode


@dataclass(slots=True)
class _Node(WorkflowNode):
    node_id: str
    dependencies: tuple[str, ...]

    input_schema: Mapping[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "required": ["dependency_results"],
            "properties": {"dependency_results": {"type": "object"}},
            "additionalProperties": True,
        }
    )
    output_schema: Mapping[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
            "additionalProperties": False,
        }
    )

    def run(self, context: Mapping[str, object]) -> Mapping[str, object]:
        dependency_results = context["dependency_results"]
        if self.node_id == "a":
            return {"value": 1}
        previous_value = dependency_results["a"]["output"]["value"]
        return {"value": int(previous_value) + 1}


def main() -> None:
    orchestrator = design_research_agents.SequentialOrchestrator()
    nodes = [
        _Node(node_id="a", dependencies=()),
        _Node(node_id="b", dependencies=("a",)),
    ]
    result = orchestrator.run(nodes)
    print(result.asdict())


if __name__ == "__main__":
    main()
