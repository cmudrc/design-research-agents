"""Shared types for built-in engineering knowledge profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphNodeRecord,
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


__all__ = [
    "KnowledgeProfile",
    "KnowledgeProfileSeedResult",
]
