"""Runnable example for ``DagOrchestrator`` with router-style branching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import design_research_agents
from design_research_agents.contracts.orchestrator import WorkflowNode


@dataclass(slots=True)
class _Node(WorkflowNode):
    node_id: str
    dependencies: tuple[str, ...]
    route_map: Mapping[str, tuple[str, ...]] | None = None

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
        if self.node_id == "router":
            return {"value": 1, "route": "left"}
        if self.node_id == "left":
            return {"value": 2}
        if self.node_id == "right":
            return {"value": 3}
        return {"value": 4}


def main() -> None:
    router_output_schema = {
        "type": "object",
        "required": ["value", "route"],
        "properties": {
            "value": {"type": "integer"},
            "route": {"type": "string"},
        },
        "additionalProperties": False,
    }

    nodes: list[_Node] = [
        _Node(
            node_id="router",
            dependencies=(),
            route_map={"left": ("left",), "right": ("right",)},
            output_schema=router_output_schema,
        ),
        _Node(node_id="left", dependencies=("router",)),
        _Node(node_id="right", dependencies=("router",)),
        _Node(node_id="merge", dependencies=("left",)),
    ]

    orchestrator = design_research_agents.DagOrchestrator()
    result = orchestrator.run(nodes, failure_policy="propagate_failed_state")
    print(result.asdict())


if __name__ == "__main__":
    main()
