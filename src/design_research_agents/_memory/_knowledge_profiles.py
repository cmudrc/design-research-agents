"""Built-in engineering knowledge profiles for memory and graph stores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphMemoryStore,
    GraphNodeRecord,
    MemoryStore,
    MemoryWriteRecord,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class KnowledgeProfile:
    """Built-in deterministic knowledge profile."""

    name: str
    """Stable profile name."""
    description: str
    """Human-readable profile summary."""
    records: tuple[MemoryWriteRecord, ...]
    """Vector/text memory records included in the profile."""
    graph_nodes: tuple[GraphNodeRecord, ...] = ()
    """Graph nodes included in the profile."""
    graph_edges: tuple[GraphEdgeRecord, ...] = ()
    """Graph edges included in the profile."""
    sources: tuple[str, ...] = ()
    """Short source labels for profile provenance."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "records": [record.to_dict() for record in self.records],
            "graph_nodes": [node.to_dict() for node in self.graph_nodes],
            "graph_edges": [edge.to_dict() for edge in self.graph_edges],
            "sources": list(self.sources),
        }


@dataclass(slots=True, frozen=True, kw_only=True)
class KnowledgeProfileSeedResult:
    """Summary of seeding one built-in knowledge profile into configured stores."""

    profile_name: str
    """Seeded profile name."""
    namespace: str
    """Namespace the profile was written into."""
    memory_records_written: int
    """Number of vector/text memory records written."""
    graph_nodes_written: int
    """Number of graph nodes written."""
    graph_edges_written: int
    """Number of graph edges written."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


_BUILTIN_KNOWLEDGE_PROFILES: dict[str, KnowledgeProfile] = {
    "stem": KnowledgeProfile(
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
    ),
    "aerospace": KnowledgeProfile(
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
    ),
    "mechanics": KnowledgeProfile(
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
    ),
}


def list_builtin_knowledge_profiles() -> tuple[str, ...]:
    """Return available built-in knowledge profile names."""
    return tuple(sorted(_BUILTIN_KNOWLEDGE_PROFILES.keys()))


def load_builtin_knowledge_profile(profile_name: str) -> KnowledgeProfile:
    """Return one built-in knowledge profile by name.

    Args:
        profile_name: Built-in profile name.

    Returns:
        Built-in knowledge profile.

    Raises:
        ValueError: If the requested profile is unknown.
    """
    normalized_name = profile_name.strip().lower()
    profile = _BUILTIN_KNOWLEDGE_PROFILES.get(normalized_name)
    if profile is None:
        available = ", ".join(list_builtin_knowledge_profiles())
        raise ValueError(f"Unknown knowledge profile '{profile_name}'. Available profiles: {available}.")
    return profile


def seed_builtin_knowledge_profile(
    profile_name: str,
    *,
    memory_store: MemoryStore | None = None,
    graph_store: GraphMemoryStore | None = None,
    namespace: str = "default",
) -> KnowledgeProfileSeedResult:
    """Seed one built-in profile into configured memory stores.

    Args:
        profile_name: Built-in profile name.
        memory_store: Optional text/vector memory store.
        graph_store: Optional graph memory store.
        namespace: Namespace used for writes.

    Returns:
        Summary of records written.

    Raises:
        ValueError: If both ``memory_store`` and ``graph_store`` are missing.
    """
    if memory_store is None and graph_store is None:
        raise ValueError("At least one of memory_store or graph_store must be provided.")

    profile = load_builtin_knowledge_profile(profile_name)
    memory_records_written = 0
    graph_nodes_written = 0
    graph_edges_written = 0

    if memory_store is not None:
        memory_records_written = len(memory_store.write(list(profile.records), namespace=namespace))
    if graph_store is not None:
        graph_nodes_written = len(graph_store.upsert_nodes(list(profile.graph_nodes), namespace=namespace))
        graph_edges_written = len(graph_store.upsert_edges(list(profile.graph_edges), namespace=namespace))

    return KnowledgeProfileSeedResult(
        profile_name=profile.name,
        namespace=namespace.strip() or "default",
        memory_records_written=memory_records_written,
        graph_nodes_written=graph_nodes_written,
        graph_edges_written=graph_edges_written,
    )


def iter_builtin_knowledge_profiles() -> Sequence[KnowledgeProfile]:
    """Return built-in knowledge profiles in deterministic name order."""
    return tuple(_BUILTIN_KNOWLEDGE_PROFILES[name] for name in list_builtin_knowledge_profiles())


__all__ = [
    "KnowledgeProfile",
    "KnowledgeProfileSeedResult",
    "iter_builtin_knowledge_profiles",
    "list_builtin_knowledge_profiles",
    "load_builtin_knowledge_profile",
    "seed_builtin_knowledge_profile",
]
