"""Built-in mechanics knowledge profile."""

from __future__ import annotations

from design_research_agents._contracts._memory import GraphEdgeRecord, GraphNodeRecord, MemoryWriteRecord
from design_research_agents._memory._knowledge_profile_types import KnowledgeProfile

PROFILE = KnowledgeProfile(
    name="mechanics",
    description="Mechanics formulas and material properties for physical design work.",
    records=(
        MemoryWriteRecord(
            content=(
                "Hooke's law for a linear spring: F = k * x, where F is restoring force, "
                "k is spring constant, and x is displacement."
            ),
            metadata={"profile": "mechanics", "kind": "formula", "topic": "spring"},
        ),
        MemoryWriteRecord(
            content=(
                "Euler-Bernoulli beam bending stress relation: sigma = M * c / I, where M is bending moment, "
                "c is distance from the neutral axis, and I is second moment of area."
            ),
            metadata={"profile": "mechanics", "kind": "formula", "topic": "beam_bending"},
        ),
        MemoryWriteRecord(
            content=(
                "Steel reference properties: density about 7850 kg/m^3 and Young's modulus about 200 GPa. "
                "These are good baseline values for early trade studies."
            ),
            metadata={"profile": "mechanics", "kind": "material_property", "topic": "steel"},
        ),
        MemoryWriteRecord(
            content=(
                "Aluminum 6061-T6 reference properties: density about 2700 kg/m^3, "
                "Young's modulus about 69 GPa, and yield strength about 276 MPa."
            ),
            metadata={"profile": "mechanics", "kind": "material_property", "topic": "aluminum_6061_t6"},
        ),
    ),
    graph_nodes=(
        GraphNodeRecord(node_id="hookes-law", name="Hooke's Law", node_type="formula"),
        GraphNodeRecord(node_id="spring-force", name="Spring Force", node_type="quantity"),
        GraphNodeRecord(node_id="spring-constant", name="Spring Constant", node_type="quantity"),
        GraphNodeRecord(node_id="beam-bending-stress", name="Beam Bending Stress", node_type="formula"),
        GraphNodeRecord(node_id="bending-moment", name="Bending Moment", node_type="quantity"),
        GraphNodeRecord(node_id="steel", name="Steel", node_type="material"),
        GraphNodeRecord(node_id="aluminum-6061-t6", name="Aluminum 6061-T6", node_type="material"),
    ),
    graph_edges=(
        GraphEdgeRecord(source_id="hookes-law", target_id="spring-force", relationship="defines"),
        GraphEdgeRecord(source_id="hookes-law", target_id="spring-constant", relationship="depends_on"),
        GraphEdgeRecord(source_id="beam-bending-stress", target_id="bending-moment", relationship="depends_on"),
        GraphEdgeRecord(source_id="aluminum-6061-t6", target_id="beam-bending-stress", relationship="used_with"),
        GraphEdgeRecord(source_id="steel", target_id="beam-bending-stress", relationship="used_with"),
    ),
    sources=("Introductory mechanics references", "Common material-property baselines"),
)


__all__ = ["PROFILE"]
