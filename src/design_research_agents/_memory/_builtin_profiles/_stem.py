"""Built-in STEM knowledge profile."""

from __future__ import annotations

from design_research_agents._contracts._memory import GraphEdgeRecord, GraphNodeRecord, MemoryWriteRecord
from design_research_agents._memory._knowledge_profile_types import KnowledgeProfile

PROFILE = KnowledgeProfile(
    name="stem",
    description="Foundational STEM formulas and unit-aware reference facts.",
    records=(
        MemoryWriteRecord(
            content=(
                "Density relation: rho = m / V, where rho is density, m is mass, and V is volume. "
                "Use SI units consistently to avoid scale errors."
            ),
            metadata={"profile": "stem", "kind": "formula", "topic": "density"},
        ),
        MemoryWriteRecord(
            content=(
                "Ideal gas law: p V = n R T. For engineering calculations, keep pressure, temperature, "
                "and gas constant units consistent."
            ),
            metadata={"profile": "stem", "kind": "formula", "topic": "thermodynamics"},
        ),
        MemoryWriteRecord(
            content=(
                "Quadratic formula: x = (-b +/- sqrt(b^2 - 4 a c)) / (2 a). "
                "It is useful when closed-form polynomial roots are needed."
            ),
            metadata={"profile": "stem", "kind": "formula", "topic": "algebra"},
        ),
    ),
    graph_nodes=(
        GraphNodeRecord(node_id="density", name="Density", node_type="formula"),
        GraphNodeRecord(node_id="mass", name="Mass", node_type="quantity"),
        GraphNodeRecord(node_id="volume", name="Volume", node_type="quantity"),
        GraphNodeRecord(node_id="ideal-gas-law", name="Ideal Gas Law", node_type="formula"),
        GraphNodeRecord(node_id="pressure", name="Pressure", node_type="quantity"),
        GraphNodeRecord(node_id="temperature", name="Temperature", node_type="quantity"),
    ),
    graph_edges=(
        GraphEdgeRecord(source_id="density", target_id="mass", relationship="depends_on"),
        GraphEdgeRecord(source_id="density", target_id="volume", relationship="depends_on"),
        GraphEdgeRecord(source_id="ideal-gas-law", target_id="pressure", relationship="depends_on"),
        GraphEdgeRecord(source_id="ideal-gas-law", target_id="temperature", relationship="depends_on"),
    ),
    sources=("Foundational engineering formulas",),
)


__all__ = ["PROFILE"]
