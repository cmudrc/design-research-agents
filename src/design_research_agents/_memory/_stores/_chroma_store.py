"""ChromaDB-backed memory store with lexical and vector retrieval."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from design_research_agents._contracts._memory import (
    MemoryRecord,
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
)
from design_research_agents._memory._embedding import EmbeddingProvider

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def _default_chroma_path() -> Path:
    """Return deterministic default Chroma persistence directory."""
    return Path.cwd() / "artifacts" / "memory" / "chroma"


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _normalize_namespace(namespace: str) -> str:
    """Normalize namespace to a non-empty default value."""
    normalized = namespace.strip()
    return normalized or "default"


def _tokenize(text: str) -> list[str]:
    """Tokenize text for lexical overlap scoring."""
    return _TOKEN_PATTERN.findall(text.lower())


def _lexical_score(*, query_tokens: Sequence[str], content: str) -> float:
    """Compute a deterministic lexical overlap score in ``[0, 1]``."""
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


def _metadata_matches(*, metadata: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    """Return whether metadata satisfies exact-match filter constraints."""
    return all(metadata.get(key) == expected_value for key, expected_value in filters.items())


def _distance_to_similarity(distance: object) -> float | None:
    """Convert a Chroma distance payload into a normalized similarity."""
    if not isinstance(distance, (int, float)):
        return None
    return 1.0 / (1.0 + float(distance))


def _parse_stored_metadata(raw_metadata: object) -> dict[str, object]:
    """Parse stored metadata payload back into the original metadata mapping."""
    if not isinstance(raw_metadata, Mapping):
        return {}
    metadata_json = raw_metadata.get("metadata_json")
    if not isinstance(metadata_json, str):
        return {}
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return {}


class ChromaMemoryStore(MemoryStore):
    """Persistent Chroma-backed memory store with optional vector retrieval."""

    def __init__(
        self,
        *,
        persist_directory: str | Path | None = None,
        collection_name: str = "design_research_agents_memory",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model: str | None = None,
        client: Any | None = None,
    ) -> None:
        """Initialize a Chroma-backed memory store.

        Args:
            persist_directory: Optional Chroma persistence directory.
            collection_name: Collection name shared across namespaces.
            embedding_provider: Optional embedding provider for vector retrieval.
            embedding_model: Optional model key used for embedding rows.
            client: Optional injected Chroma client for testing.

        Raises:
            ImportError: If ``chromadb`` is unavailable.
        """
        resolved_directory = Path(persist_directory) if persist_directory is not None else _default_chroma_path()
        resolved_directory.mkdir(parents=True, exist_ok=True)
        self._persist_directory = resolved_directory
        self._embedding_provider = embedding_provider
        model_name = (embedding_model or "").strip()
        if not model_name and embedding_provider is not None:
            model_name = embedding_provider.model_name
        self._embedding_model = model_name or "default"

        if client is None:
            try:
                chromadb_module = import_module("chromadb")
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "ChromaMemoryStore requires the optional 'chromadb' dependency. "
                    'Install it with `pip install -e ".[memory_chroma]"`.'
                ) from exc
            client = chromadb_module.PersistentClient(path=str(resolved_directory))
        self._client = client
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @property
    def persist_directory(self) -> Path:
        """Return persistence directory used by this store."""
        return self._persist_directory

    def close(self) -> None:
        """Release any store-owned resources."""
        close_callable = getattr(self._client, "close", None)
        if callable(close_callable):
            close_callable()

    def write(
        self,
        records: Sequence[MemoryWriteRecord],
        *,
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        """Persist records and return normalized stored records."""
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
        existing_records = self._fetch_records_by_ids(item_ids)
        existing_by_id = {record.item_id: record for record in existing_records}

        embeddings: list[list[float]] | None = None
        if self._embedding_provider is not None:
            embeddings = self._embedding_provider.embed([record.content for record in normalized_records])

        metadatas: list[dict[str, object]] = []
        documents: list[str] = []
        upsert_ids: list[str] = []
        upsert_embeddings: list[list[float]] | None = [] if embeddings is not None else None
        for index, record in enumerate(normalized_records):
            item_id = item_ids[index]
            existing = existing_by_id.get(item_id)
            created_at = (existing.created_at or timestamp) if existing is not None else timestamp
            metadata_payload = _build_storage_metadata(
                namespace=normalized_namespace,
                metadata=dict(record.metadata),
                created_at=created_at,
                updated_at=timestamp,
            )
            metadatas.append(metadata_payload)
            documents.append(record.content)
            upsert_ids.append(item_id)
            if embeddings is not None and upsert_embeddings is not None:
                vector = embeddings[index] if index < len(embeddings) else None
                if vector is None:
                    upsert_embeddings = None
                elif upsert_embeddings is not None:
                    upsert_embeddings.append(vector)

        upsert_kwargs: dict[str, object] = {
            "ids": upsert_ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if upsert_embeddings is not None:
            upsert_kwargs["embeddings"] = upsert_embeddings
        self._collection.upsert(**upsert_kwargs)
        return self._fetch_records_by_ids(item_ids)

    def search(self, query: MemorySearchQuery) -> list[MemoryRecord]:
        """Search records using lexical relevance and optional vector similarity."""
        normalized_namespace = _normalize_namespace(query.namespace)
        top_k = max(1, int(query.top_k))
        metadata_filters = dict(query.metadata_filters)
        query_tokens = _tokenize(str(query.text))

        query_vector: list[float] | None = None
        if self._embedding_provider is not None:
            embedded = self._embedding_provider.embed([query.text])
            if embedded and embedded[0]:
                query_vector = embedded[0]

        if query_vector is not None:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=max(top_k * 4, top_k),
                where=_build_where_clause(namespace=normalized_namespace, metadata_filters=metadata_filters),
                include=["documents", "metadatas", "distances"],
            )
            candidate_records = _records_from_query_results(results)
        else:
            candidate_records = self._records_from_get(
                self._collection.get(
                    where={"namespace": normalized_namespace},
                    include=["documents", "metadatas"],
                )
            )

        scored_records: list[MemoryRecord] = []
        for candidate in candidate_records:
            if candidate.namespace != normalized_namespace:
                continue
            if not _metadata_matches(metadata=candidate.metadata, filters=metadata_filters):
                continue
            lexical_score = _lexical_score(query_tokens=query_tokens, content=candidate.content)
            vector_score = candidate.vector_score
            final_score = lexical_score if vector_score is None else 0.7 * vector_score + 0.3 * lexical_score
            if query.min_score is not None and final_score < float(query.min_score):
                continue
            scored_records.append(
                MemoryRecord(
                    item_id=candidate.item_id,
                    namespace=candidate.namespace,
                    content=candidate.content,
                    metadata=candidate.metadata,
                    created_at=candidate.created_at,
                    updated_at=candidate.updated_at,
                    score=round(final_score, 8),
                    lexical_score=round(lexical_score, 8),
                    vector_score=None if vector_score is None else round(vector_score, 8),
                )
            )

        scored_records.sort(
            key=lambda record: (
                float(record.score or 0.0),
                str(record.updated_at or ""),
                record.item_id,
            ),
            reverse=True,
        )
        return scored_records[:top_k]

    def _fetch_records_by_ids(self, item_ids: Sequence[str]) -> list[MemoryRecord]:
        """Fetch records in caller-provided id order."""
        if not item_ids:
            return []
        results = self._collection.get(
            ids=list(item_ids),
            include=["documents", "metadatas"],
        )
        records_by_id = {record.item_id: record for record in self._records_from_get(results)}
        ordered_records: list[MemoryRecord] = []
        for item_id in item_ids:
            record = records_by_id.get(item_id)
            if record is not None:
                ordered_records.append(record)
        return ordered_records

    def _records_from_get(self, payload: object) -> list[MemoryRecord]:
        """Convert ``collection.get`` payloads into normalized memory records."""
        if not isinstance(payload, Mapping):
            return []

        ids = payload.get("ids")
        documents = payload.get("documents")
        metadatas = payload.get("metadatas")
        if not isinstance(ids, list) or not isinstance(documents, list) or not isinstance(metadatas, list):
            return []

        records: list[MemoryRecord] = []
        for item_id, document, raw_metadata in zip(ids, documents, metadatas, strict=False):
            metadata_dict = _parse_stored_metadata(raw_metadata)
            records.append(
                MemoryRecord(
                    item_id=str(item_id),
                    namespace=str(_safe_mapping_get(raw_metadata, "namespace", "default")),
                    content=str(document),
                    metadata=metadata_dict,
                    created_at=_optional_string(_safe_mapping_get(raw_metadata, "created_at")),
                    updated_at=_optional_string(_safe_mapping_get(raw_metadata, "updated_at")),
                )
            )
        return records


def _records_from_query_results(payload: object) -> list[MemoryRecord]:
    """Convert ``collection.query`` payloads into normalized memory records."""
    if not isinstance(payload, Mapping):
        return []

    ids = payload.get("ids")
    documents = payload.get("documents")
    metadatas = payload.get("metadatas")
    distances = payload.get("distances")
    if not isinstance(ids, list) or not isinstance(documents, list) or not isinstance(metadatas, list):
        return []
    if not ids:
        return []

    first_ids = ids[0] if ids and isinstance(ids[0], list) else []
    first_documents = documents[0] if documents and isinstance(documents[0], list) else []
    first_metadatas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else []
    first_distances = distances[0] if distances and isinstance(distances[0], list) else []

    records: list[MemoryRecord] = []
    for item_id, document, raw_metadata, distance in zip(
        first_ids,
        first_documents,
        first_metadatas,
        first_distances,
        strict=False,
    ):
        metadata_dict = _parse_stored_metadata(raw_metadata)
        records.append(
            MemoryRecord(
                item_id=str(item_id),
                namespace=str(_safe_mapping_get(raw_metadata, "namespace", "default")),
                content=str(document),
                metadata=metadata_dict,
                created_at=_optional_string(_safe_mapping_get(raw_metadata, "created_at")),
                updated_at=_optional_string(_safe_mapping_get(raw_metadata, "updated_at")),
                vector_score=_distance_to_similarity(distance),
            )
        )
    return records


def _safe_mapping_get(mapping: object, key: str, default: object | None = None) -> object | None:
    """Read one key from a mapping-like object with deterministic fallback."""
    if not isinstance(mapping, Mapping):
        return default
    return cast(object | None, mapping.get(key, default))


def _optional_string(value: object) -> str | None:
    """Return a trimmed string when ``value`` is string-like, otherwise ``None``."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _build_storage_metadata(
    *,
    namespace: str,
    metadata: Mapping[str, object],
    created_at: str,
    updated_at: str,
) -> dict[str, object]:
    """Build one Chroma metadata payload for a stored memory record."""
    payload: dict[str, object] = {
        "namespace": namespace,
        "created_at": created_at,
        "updated_at": updated_at,
        "metadata_json": json.dumps(dict(metadata), ensure_ascii=True, sort_keys=True),
    }
    for key, value in metadata.items():
        if isinstance(value, bool | int | float | str):
            payload[f"meta__{key}"] = value
    return payload


def _build_where_clause(
    *,
    namespace: str,
    metadata_filters: Mapping[str, object],
) -> dict[str, object]:
    """Build one best-effort Chroma ``where`` clause."""
    clauses: list[dict[str, object]] = [{"namespace": namespace}]
    for key, value in metadata_filters.items():
        if isinstance(value, bool | int | float | str):
            clauses.append({f"meta__{key}": value})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


__all__ = ["ChromaMemoryStore"]
