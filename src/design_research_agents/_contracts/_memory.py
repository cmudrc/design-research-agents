"""Memory and graph-memory contracts for pluggable persistence backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self


@dataclass(slots=True, frozen=True, kw_only=True)
class MemoryWriteRecord:
    """One record payload to be persisted into a memory store."""

    content: str
    """Primary text content to persist."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Optional metadata fields used for downstream filtering/ranking."""
    item_id: str | None = None
    """Optional caller-supplied id for deterministic upserts."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Serialized dataclass payload.
        """
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class MemorySearchQuery:
    """Structured memory search query."""

    text: str
    """Natural language query text."""
    namespace: str = "default"
    """Namespace partition used for isolation."""
    top_k: int = 5
    """Maximum number of matches to return."""
    min_score: float | None = None
    """Optional minimum score threshold for returned matches."""
    metadata_filters: dict[str, object] = field(default_factory=dict)
    """Exact-match metadata filters applied before ranking."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Serialized dataclass payload.
        """
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class MemoryRecord:
    """Retrieved or persisted memory record."""

    item_id: str
    """Stable memory record identifier."""
    namespace: str
    """Namespace partition that owns this record."""
    content: str
    """Record content text."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Record metadata fields."""
    created_at: str | None = None
    """ISO timestamp for record creation."""
    updated_at: str | None = None
    """ISO timestamp for last record update."""
    score: float | None = None
    """Combined ranking score when returned by search."""
    lexical_score: float | None = None
    """Lexical relevance score."""
    vector_score: float | None = None
    """Vector similarity score when embeddings are available."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Serialized dataclass payload.
        """
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class GraphNodeRecord:
    """One graph node stored in a graph-memory backend."""

    node_id: str
    """Stable node identifier."""
    name: str
    """Human-readable node label."""
    node_type: str = "entity"
    """Coarse node type such as ``component`` or ``formula``."""
    description: str = ""
    """Optional free-text node description."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Optional node metadata fields."""
    created_at: str | None = None
    """ISO timestamp for node creation."""
    updated_at: str | None = None
    """ISO timestamp for last node update."""
    score: float | None = None
    """Optional retrieval score when returned by a graph query."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Serialized dataclass payload.
        """
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class GraphEdgeRecord:
    """One graph edge stored in a graph-memory backend."""

    source_id: str
    """Source node identifier."""
    target_id: str
    """Target node identifier."""
    relationship: str
    """Normalized relationship label."""
    edge_id: str | None = None
    """Stable edge identifier; derived by stores when omitted."""
    metadata: dict[str, object] = field(default_factory=dict)
    """Optional edge metadata fields."""
    created_at: str | None = None
    """ISO timestamp for edge creation."""
    updated_at: str | None = None
    """ISO timestamp for last edge update."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Serialized dataclass payload.
        """
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class GraphSearchQuery:
    """Structured graph-memory search query."""

    text: str
    """Natural language query text."""
    namespace: str = "default"
    """Namespace partition used for isolation."""
    top_k: int = 3
    """Maximum number of seed nodes to retrieve before expansion."""
    max_hops: int = 1
    """Maximum graph-traversal distance from matched seed nodes."""
    min_score: float | None = None
    """Optional minimum score threshold for matched seed nodes."""
    metadata_filters: dict[str, object] = field(default_factory=dict)
    """Exact-match metadata filters applied to candidate nodes."""
    node_type_filters: tuple[str, ...] = ()
    """Optional allowed node types."""
    relationship_filters: tuple[str, ...] = ()
    """Optional allowed relationship labels for returned edges."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Serialized dataclass payload.
        """
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class GraphSubgraphResult:
    """One retrieved graph subgraph."""

    namespace: str
    """Namespace partition queried."""
    query_text: str
    """Original query text."""
    matched_node_ids: tuple[str, ...] = ()
    """Seed node ids that matched the text query before graph expansion."""
    nodes: tuple[GraphNodeRecord, ...] = ()
    """Retrieved nodes, including traversal context."""
    edges: tuple[GraphEdgeRecord, ...] = ()
    """Retrieved edges connecting included nodes."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation.

        Returns:
            Serialized dataclass payload.
        """
        return {
            "namespace": self.namespace,
            "query_text": self.query_text,
            "matched_node_ids": list(self.matched_node_ids),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


class MemoryStore(Protocol):
    """Protocol implemented by memory stores used by workflows and agents."""

    def write(
        self,
        records: Sequence[MemoryWriteRecord],
        *,
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        """Persist one or more records and return normalized stored entries.

        Args:
            records: Record payloads to persist.
            namespace: Namespace partition to store records under.

        Returns:
            Stored records including ids and timestamps.
        """

    def search(self, query: MemorySearchQuery) -> list[MemoryRecord]:
        """Search memory records using lexical/vector relevance.

        Args:
            query: Structured memory search query.

        Returns:
            Ordered list of matching records.
        """

    def close(self) -> None:
        """Release any store-owned resources.

        Implementations that do not own external resources may implement this
        as a no-op so callers can use a uniform lifecycle pattern.
        """
        return None

    def __enter__(self) -> Self:
        """Return this store for use in a ``with`` statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Close the store when exiting a ``with`` block."""
        del exc_type, exc, tb
        self.close()
        return None


class GraphMemoryStore(Protocol):
    """Protocol implemented by graph-memory stores used by agents and patterns."""

    def upsert_nodes(
        self,
        nodes: Sequence[GraphNodeRecord],
        *,
        namespace: str = "default",
    ) -> list[GraphNodeRecord]:
        """Persist graph nodes and return normalized stored entries.

        Args:
            nodes: Node payloads to persist.
            namespace: Namespace partition to store nodes under.

        Returns:
            Stored nodes including timestamps.
        """

    def upsert_edges(
        self,
        edges: Sequence[GraphEdgeRecord],
        *,
        namespace: str = "default",
    ) -> list[GraphEdgeRecord]:
        """Persist graph edges and return normalized stored entries.

        Args:
            edges: Edge payloads to persist.
            namespace: Namespace partition to store edges under.

        Returns:
            Stored edges including ids and timestamps.
        """

    def query_subgraph(self, query: GraphSearchQuery) -> GraphSubgraphResult:
        """Retrieve a relevant graph subgraph for one structured query.

        Args:
            query: Structured graph-memory search query.

        Returns:
            Retrieved subgraph result.
        """

    def close(self) -> None:
        """Release any store-owned resources."""
        return None

    def __enter__(self) -> Self:
        """Return this store for use in a ``with`` statement."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Close the store when exiting a ``with`` block."""
        del exc_type, exc, tb
        self.close()
        return None


__all__ = [
    "GraphEdgeRecord",
    "GraphMemoryStore",
    "GraphNodeRecord",
    "GraphSearchQuery",
    "GraphSubgraphResult",
    "MemoryRecord",
    "MemorySearchQuery",
    "MemoryStore",
    "MemoryWriteRecord",
]
