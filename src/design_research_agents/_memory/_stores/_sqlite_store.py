"""SQLite-backed memory store with lexical and optional vector retrieval."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from design_research_agents._contracts._memory import (
    MemoryRecord,
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
)
from design_research_agents._memory._embedding import EmbeddingProvider

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def _default_memory_path() -> Path:
    """Return deterministic default sqlite path under ``artifacts/memory``.

    Returns:
        Default sqlite database file path.
    """
    return Path.cwd() / "artifacts" / "memory" / "memory.sqlite3"


class SQLiteMemoryStore(MemoryStore):
    """Persistent local memory store with optional embedding-based retrieval."""

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize a local sqlite memory store.

        Args:
            db_path: Optional sqlite database file path.
            embedding_provider: Optional embedding provider for vector retrieval.
            embedding_model: Optional model key used for embedding rows.
        """
        resolved_path = Path(db_path) if db_path is not None else _default_memory_path()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved_path
        self._embedding_provider = embedding_provider
        model_name = (embedding_model or "").strip()
        if not model_name and embedding_provider is not None:
            model_name = embedding_provider.model_name
        # Keep one stable embedding model tag per store so search joins hit the intended vectors.
        self._embedding_model = model_name or "default"
        self._connection = sqlite3.connect(str(self._db_path))
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    @property
    def db_path(self) -> Path:
        """Return sqlite database path used by this store.

        Returns:
            Configured sqlite database path.
        """
        return self._db_path

    def close(self) -> None:
        """Close sqlite connection."""
        self._connection.close()

    def write(
        self,
        records: Sequence[MemoryWriteRecord],
        *,
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        """Persist records and return normalized stored records.

        Args:
            records: Memory record payloads to persist.
            namespace: Namespace partition for writes.

        Returns:
            Stored records in caller-provided order.
        """
        normalized_namespace = _normalize_namespace(namespace)
        if not records:
            return []

        timestamp = _utc_now_iso()
        normalized_records = [
            MemoryWriteRecord(
                content=str(record.content),
                metadata=dict(record.metadata),
                item_id=str(record.item_id).strip() if record.item_id else None,
            )
            for record in records
        ]
        item_ids = [record.item_id or uuid4().hex for record in normalized_records]

        embeddings: list[list[float]] | None = None
        if self._embedding_provider is not None:
            # Embed in batch to amortize provider overhead and keep row/vector alignment by index.
            embeddings = self._embedding_provider.embed([record.content for record in normalized_records])

        with self._connection:
            for index, record in enumerate(normalized_records):
                item_id = item_ids[index]
                metadata_json = json.dumps(record.metadata, ensure_ascii=True, sort_keys=True)
                self._connection.execute(
                    """
                    INSERT INTO memory_items (
                        id,
                        namespace,
                        content,
                        metadata_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        namespace=excluded.namespace,
                        content=excluded.content,
                        metadata_json=excluded.metadata_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        item_id,
                        normalized_namespace,
                        record.content,
                        metadata_json,
                        timestamp,
                        timestamp,
                    ),
                )

                if embeddings is None:
                    continue
                vector = embeddings[index] if index < len(embeddings) else None
                if vector is None:
                    continue
                self._connection.execute(
                    """
                    INSERT INTO memory_embeddings (item_id, model, vector_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(item_id, model) DO UPDATE SET
                        vector_json=excluded.vector_json
                    """,
                    (
                        item_id,
                        self._embedding_model,
                        json.dumps(vector, ensure_ascii=True),
                    ),
                )

        return self._fetch_records_by_ids(item_ids)

    def search(self, query: MemorySearchQuery) -> list[MemoryRecord]:
        """Search records using lexical relevance and optional vector similarity.

        Args:
            query: Structured memory search query.

        Returns:
            Ranked memory records matching the query.
        """
        normalized_namespace = _normalize_namespace(query.namespace)
        top_k = max(1, int(query.top_k))
        metadata_filters = dict(query.metadata_filters)

        query_text = str(query.text)
        query_tokens = _tokenize(query_text)
        query_vector: list[float] | None = None
        if self._embedding_provider is not None:
            embedded = self._embedding_provider.embed([query_text])
            if embedded and embedded[0]:
                query_vector = embedded[0]

        if self._embedding_provider is not None:
            rows = self._connection.execute(
                """
                SELECT
                    i.id,
                    i.namespace,
                    i.content,
                    i.metadata_json,
                    i.created_at,
                    i.updated_at,
                    e.vector_json
                FROM memory_items AS i
                LEFT JOIN memory_embeddings AS e
                    ON e.item_id = i.id AND e.model = ?
                WHERE i.namespace = ?
                """,
                (self._embedding_model, normalized_namespace),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT
                    i.id,
                    i.namespace,
                    i.content,
                    i.metadata_json,
                    i.created_at,
                    i.updated_at,
                    NULL AS vector_json
                FROM memory_items AS i
                WHERE i.namespace = ?
                """,
                (normalized_namespace,),
            ).fetchall()

        scored_records: list[MemoryRecord] = []
        for row in rows:
            metadata = _parse_metadata_json(row["metadata_json"])
            if not _metadata_matches(metadata=metadata, filters=metadata_filters):
                continue

            content = str(row["content"])
            lexical_score = _lexical_score(query_tokens=query_tokens, content=content)
            vector_score: float | None = None
            vector_json = row["vector_json"]
            if query_vector is not None and isinstance(vector_json, str):
                vector = _parse_vector_json(vector_json)
                vector_score = _cosine_similarity(query_vector, vector)

            # Blend lexical/vector scores to preserve useful fallback behavior when vectors are noisy.
            final_score = lexical_score if vector_score is None else 0.7 * vector_score + 0.3 * lexical_score

            if query.min_score is not None and final_score < float(query.min_score):
                continue

            scored_records.append(
                MemoryRecord(
                    item_id=str(row["id"]),
                    namespace=str(row["namespace"]),
                    content=content,
                    metadata=metadata,
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                    score=round(final_score, 8),
                    lexical_score=round(lexical_score, 8),
                    vector_score=(None if vector_score is None else round(vector_score, 8)),
                )
            )

        scored_records.sort(
            # Secondary keys keep ordering deterministic for ties across identical scores.
            key=lambda record: (
                float(record.score or 0.0),
                str(record.updated_at or ""),
                record.item_id,
            ),
            reverse=True,
        )
        return scored_records[:top_k]

    def _fetch_records_by_ids(self, item_ids: Sequence[str]) -> list[MemoryRecord]:
        """Fetch records in caller-provided id order.

        Args:
            item_ids: Memory item ids to load.

        Returns:
            Stored records in requested id order.
        """
        records_by_id: dict[str, MemoryRecord] = {}
        for item_id in item_ids:
            row = self._connection.execute(
                """
                SELECT id, namespace, content, metadata_json, created_at, updated_at
                FROM memory_items
                WHERE id = ?
                """,
                (item_id,),
            ).fetchone()
            if row is None:
                continue
            records_by_id[item_id] = MemoryRecord(
                item_id=str(row["id"]),
                namespace=str(row["namespace"]),
                content=str(row["content"]),
                metadata=_parse_metadata_json(row["metadata_json"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
            )

        ordered_records: list[MemoryRecord] = []
        for item_id in item_ids:
            record = records_by_id.get(item_id)
            if record is not None:
                ordered_records.append(record)
        return ordered_records

    def _initialize_schema(self) -> None:
        """Create required tables and indexes if they are missing."""
        with self._connection:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """)
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    item_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY(item_id, model)
                )
                """)
            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_items_namespace
                ON memory_items(namespace)
                """)


def _metadata_matches(*, metadata: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    """Return whether metadata satisfies exact-match filter constraints.

    Args:
        metadata: Metadata dictionary to validate.
        filters: Required exact-match filter pairs.

    Returns:
        ``True`` when all requested filters match.
    """
    return all(metadata.get(key) == expected_value for key, expected_value in filters.items())


def _parse_metadata_json(value: object) -> dict[str, object]:
    """Parse metadata JSON into a dictionary.

    Args:
        value: Raw metadata payload.

    Returns:
        Parsed metadata dictionary or empty mapping on parse failure.
    """
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {}


def _parse_vector_json(value: str) -> list[float]:
    """Parse vector JSON into a float list.

    Args:
        value: JSON-serialized vector payload.

    Returns:
        Parsed float vector, or empty list on parse failure.
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    vector: list[float] = []
    for token in parsed:
        try:
            vector.append(float(token))
        except (TypeError, ValueError):
            return []
    return vector


def _tokenize(text: str) -> list[str]:
    """Tokenize text for lexical overlap scoring.

    Args:
        text: Input text.

    Returns:
        Lower-cased token list.
    """
    return _TOKEN_PATTERN.findall(text.lower())


def _lexical_score(*, query_tokens: Sequence[str], content: str) -> float:
    """Compute lexical overlap score in [0, 1].

    Args:
        query_tokens: Tokenized query text.
        content: Candidate memory content.

    Returns:
        Lexical overlap score in the inclusive range ``[0, 1]``.
    """
    if not query_tokens:
        return 0.0

    query_set = set(query_tokens)
    content_tokens = _tokenize(content)
    if not content_tokens:
        return 0.0
    content_set = set(content_tokens)
    overlap = len(query_set.intersection(content_set))
    score = overlap / float(len(query_set))

    if query_set and " ".join(query_tokens) in content.lower():
        score = min(1.0, score + 0.15)
    return float(score)


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float | None:
    """Compute cosine similarity normalized to [0, 1].

    Args:
        vector_a: Left vector.
        vector_b: Right vector.

    Returns:
        Normalized cosine similarity, or ``None`` for incompatible vectors.
    """
    if len(vector_a) != len(vector_b):
        return None

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for value_a, value_b in zip(vector_a, vector_b, strict=True):
        dot += value_a * value_b
        norm_a += value_a * value_a
        norm_b += value_b * value_b
    if norm_a == 0.0 or norm_b == 0.0:
        return None

    cosine = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    normalized = (cosine + 1.0) / 2.0
    return max(0.0, min(1.0, normalized))


def _normalize_namespace(namespace: str) -> str:
    """Normalize namespace to non-empty default-safe value.

    Args:
        namespace: Raw namespace value.

    Returns:
        Normalized namespace.
    """
    normalized = namespace.strip()
    return normalized or "default"


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format.

    Returns:
        ISO-8601 UTC timestamp string.
    """
    return datetime.now(UTC).isoformat()


__all__ = ["SQLiteMemoryStore"]
