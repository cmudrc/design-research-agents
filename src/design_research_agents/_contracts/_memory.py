"""Memory contracts for pluggable retrieval and persistence backends."""

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


__all__ = [
    "MemoryRecord",
    "MemorySearchQuery",
    "MemoryStore",
    "MemoryWriteRecord",
]
