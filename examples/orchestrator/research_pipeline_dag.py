"""Sample research-style DAG pipeline.

Pipeline shape:
router -> retrieval -> proposer -> critic -> evaluator
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import design_research_agents
from design_research_agents.contracts.orchestrator import WorkflowNode


@dataclass(slots=True)
class _ResearchNode(WorkflowNode):
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
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        }
    )

    def run(self, context: Mapping[str, object]) -> Mapping[str, object]:
        dependency_results = context["dependency_results"]
        if self.node_id == "router":
            return {"value": "route:research"}
        if self.node_id == "retrieval":
            return {"value": "retrieved context"}
        if self.node_id == "proposer":
            source = dependency_results["retrieval"]["output"]["value"]
            return {"value": f"proposal from {source}"}
        if self.node_id == "critic":
            proposal = dependency_results["proposer"]["output"]["value"]
            return {"value": f"critique of {proposal}"}
        critique = dependency_results["critic"]["output"]["value"]
        return {"value": f"evaluation of {critique}"}


def main() -> None:
    nodes = [
        _ResearchNode(node_id="router", dependencies=()),
        _ResearchNode(node_id="retrieval", dependencies=("router",)),
        _ResearchNode(node_id="proposer", dependencies=("retrieval",)),
        _ResearchNode(node_id="critic", dependencies=("proposer",)),
        _ResearchNode(node_id="evaluator", dependencies=("critic",)),
    ]

    result = design_research_agents.DagOrchestrator().run(nodes)
    print(result.asdict())


if __name__ == "__main__":
    main()
