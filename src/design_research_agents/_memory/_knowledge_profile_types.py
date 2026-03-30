"""Shared types for deterministic engineering knowledge ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphNodeRecord,
    MemoryWriteRecord,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class KnowledgeSource:
    """One structured provenance source for ingested knowledge."""

    label: str = ""
    """Human-readable source label."""
    uri: str = ""
    """Canonical source URI."""
    kind: str = "unspecified"
    """Coarse provenance kind such as ``background_reference``."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class KnowledgeDocument:
    """One canonical knowledge document ready for ingestion."""

    document_id: str
    """Stable document identifier within one profile."""
    title: str
    """Human-readable document title."""
    content: str
    """Markdown document content to chunk and ingest."""
    sources: tuple[KnowledgeSource, ...] = ()
    """Structured provenance sources associated with the document."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class KnowledgeProfile:
    """Materialized deterministic knowledge profile."""

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
    sources: tuple[KnowledgeSource, ...] = ()
    """Structured provenance sources for the profile."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "records": [record.to_dict() for record in self.records],
            "graph_nodes": [node.to_dict() for node in self.graph_nodes],
            "graph_edges": [edge.to_dict() for edge in self.graph_edges],
            "sources": [source.to_dict() for source in self.sources],
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
    "KnowledgeDocument",
    "KnowledgeProfile",
    "KnowledgeProfileSeedResult",
    "KnowledgeSource",
]
