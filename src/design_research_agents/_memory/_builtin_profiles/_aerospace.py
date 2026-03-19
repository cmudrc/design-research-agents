"""Built-in aerospace knowledge profile."""

from __future__ import annotations

from design_research_agents._contracts._memory import GraphEdgeRecord, GraphNodeRecord, MemoryWriteRecord
from design_research_agents._memory._knowledge_profile_types import KnowledgeProfile

PROFILE = KnowledgeProfile(
    name="aerospace",
    description="Aerodynamics and aerospace-material baseline references.",
    records=(
        MemoryWriteRecord(
            content=(
                "Lift equation: L = 0.5 * rho * V^2 * S * C_L. "
                "Lift scales with dynamic pressure, reference area, and lift coefficient."
            ),
            metadata={"profile": "aerospace", "kind": "formula", "topic": "lift"},
        ),
        MemoryWriteRecord(
            content=(
                "A thrust-to-weight ratio above 1 means available thrust exceeds vehicle weight in the "
                "current gravity field, which is useful for vertical climb intuition."
            ),
            metadata={"profile": "aerospace", "kind": "heuristic", "topic": "propulsion"},
        ),
        MemoryWriteRecord(
            content=(
                "Aluminum 2024-T3 reference density is about 2780 kg/m^3 and Young's modulus is about 73 GPa. "
                "Use exact allowables from qualified material data for final design."
            ),
            metadata={"profile": "aerospace", "kind": "material_property", "topic": "aluminum_2024_t3"},
        ),
    ),
    graph_nodes=(
        GraphNodeRecord(node_id="lift-equation", name="Lift Equation", node_type="formula"),
        GraphNodeRecord(node_id="air-density", name="Air Density", node_type="quantity"),
        GraphNodeRecord(node_id="airspeed", name="Airspeed", node_type="quantity"),
        GraphNodeRecord(node_id="wing-area", name="Wing Area", node_type="quantity"),
        GraphNodeRecord(node_id="lift-coefficient", name="Lift Coefficient", node_type="quantity"),
        GraphNodeRecord(node_id="aluminum-2024-t3", name="Aluminum 2024-T3", node_type="material"),
    ),
    graph_edges=(
        GraphEdgeRecord(source_id="lift-equation", target_id="air-density", relationship="depends_on"),
        GraphEdgeRecord(source_id="lift-equation", target_id="airspeed", relationship="depends_on"),
        GraphEdgeRecord(source_id="lift-equation", target_id="wing-area", relationship="depends_on"),
        GraphEdgeRecord(source_id="lift-equation", target_id="lift-coefficient", relationship="depends_on"),
        GraphEdgeRecord(source_id="aluminum-2024-t3", target_id="lift-equation", relationship="used_with"),
    ),
    sources=("Introductory aerodynamics references", "Common aerospace material baselines"),
)


__all__ = ["PROFILE"]
