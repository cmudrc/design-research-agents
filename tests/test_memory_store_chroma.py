"""Chroma memory store contract tests."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pytest

from design_research_agents._contracts._memory import MemorySearchQuery, MemoryWriteRecord
from design_research_agents._memory import ChromaMemoryStore, EmbeddingProvider
from design_research_agents._memory._stores import _chroma_store as chroma_impl


class _StaticEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for ranking tests."""

    def __init__(self, *, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    @property
    def model_name(self) -> str:
        return "static-test-embeddings"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]] | None:
        return [list(self._vectors_by_text.get(text, [0.0, 0.0])) for text in texts]


class _FakeChromaCollection:
    """In-memory stand-in for a Chroma collection."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, object]] = {}
        self.last_upsert_embeddings: list[list[float]] | None = None

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[Mapping[str, object]],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self.last_upsert_embeddings = embeddings
        for index, item_id in enumerate(ids):
            self._records[item_id] = {
                "document": documents[index],
                "metadata": dict(metadatas[index]),
                "embedding": embeddings[index] if embeddings is not None else None,
            }

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: Mapping[str, object] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, object]:
        del include
        selected_records = self._select_records(ids=ids, where=where)
        return {
            "ids": [item_id for item_id, _record in selected_records],
            "documents": [str(record["document"]) for _item_id, record in selected_records],
            "metadatas": [dict(record["metadata"]) for _item_id, record in selected_records],
        }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: Mapping[str, object] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, object]:
        del include
        query_embedding = query_embeddings[0]
        selected_records = self._select_records(ids=None, where=where)
        ranked: list[tuple[str, dict[str, object], float]] = []
        for item_id, record in selected_records:
            embedding = record.get("embedding")
            if not isinstance(embedding, list):
                continue
            distance = _euclidean_distance(query_embedding, embedding)
            ranked.append((item_id, record, distance))
        ranked.sort(key=lambda item: item[2])
        limited = ranked[:n_results]
        return {
            "ids": [[item_id for item_id, _record, _distance in limited]],
            "documents": [[str(record["document"]) for _item_id, record, _distance in limited]],
            "metadatas": [[dict(record["metadata"]) for _item_id, record, _distance in limited]],
            "distances": [[distance for _item_id, _record, distance in limited]],
        }

    def _select_records(
        self,
        *,
        ids: list[str] | None,
        where: Mapping[str, object] | None,
    ) -> list[tuple[str, dict[str, object]]]:
        if ids is not None:
            ordered: list[tuple[str, dict[str, object]]] = []
            for item_id in ids:
                record = self._records.get(item_id)
                if record is not None:
                    ordered.append((item_id, record))
            return ordered

        selected = list(self._records.items())
        if where is None:
            return selected
        return [(item_id, record) for item_id, record in selected if _metadata_matches(record["metadata"], where)]


class _FakeChromaClient:
    """In-memory stand-in for a Chroma client."""

    def __init__(self) -> None:
        self.collection = _FakeChromaCollection()

    def get_or_create_collection(self, *, name: str) -> _FakeChromaCollection:
        del name
        return self.collection

    def close(self) -> None:
        return None


class _ShortEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that intentionally returns too few vectors."""

    @property
    def model_name(self) -> str:
        return "short-provider"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]] | None:
        del texts
        return [[1.0, 0.0]]


def test_chroma_memory_store_uses_embeddings_on_write_and_search() -> None:
    provider = _StaticEmbeddingProvider(
        vectors_by_text={
            "query": [1.0, 0.0],
            "mostly lexical": [0.0, 1.0],
            "mostly vector": [1.0, 0.0],
        }
    )
    client = _FakeChromaClient()
    store = ChromaMemoryStore(
        client=client,
        embedding_provider=provider,
    )
    store.write(
        [
            MemoryWriteRecord(content="mostly lexical"),
            MemoryWriteRecord(content="mostly vector"),
        ],
        namespace="default",
    )

    matches = store.search(MemorySearchQuery(text="query", namespace="default", top_k=2))
    store.close()

    assert client.collection.last_upsert_embeddings is not None
    assert len(client.collection.last_upsert_embeddings) == 2
    assert len(matches) == 2
    assert matches[0].content == "mostly vector"
    assert matches[0].vector_score is not None


def test_chroma_memory_store_applies_metadata_filters_without_native_where_support() -> None:
    client = _FakeChromaClient()
    store = ChromaMemoryStore(client=client)
    store.write(
        [
            MemoryWriteRecord(content="design note", metadata={"kind": "note"}),
            MemoryWriteRecord(content="design decision", metadata={"kind": "decision"}),
        ],
        namespace="workspace",
    )

    matches = store.search(
        MemorySearchQuery(
            text="design",
            namespace="workspace",
            metadata_filters={"kind": "decision"},
            top_k=5,
        )
    )
    store.close()

    assert len(matches) == 1
    assert matches[0].metadata["kind"] == "decision"


def test_chroma_store_helper_branches_and_error_paths(monkeypatch, tmp_path) -> None:
    assert chroma_impl._lexical_score(query_tokens=[], content="alpha") == 0.0
    assert chroma_impl._lexical_score(query_tokens=["alpha"], content="") == 0.0
    assert chroma_impl._distance_to_similarity("bad") is None
    assert chroma_impl._parse_stored_metadata(object()) == {}
    assert chroma_impl._parse_stored_metadata({"metadata_json": 1}) == {}
    assert chroma_impl._parse_stored_metadata({"metadata_json": "{bad"}) == {}
    assert chroma_impl._parse_stored_metadata({"metadata_json": json.dumps(["not", "a", "mapping"])}) == {}
    assert chroma_impl._safe_mapping_get(object(), "x", "fallback") == "fallback"
    assert chroma_impl._optional_string(None) is None

    storage_metadata = chroma_impl._build_storage_metadata(
        namespace="ns",
        metadata={"kind": "decision", "priority": 1, "ignored": ["list"]},
        created_at="created",
        updated_at="updated",
    )
    assert storage_metadata["meta__kind"] == "decision"
    assert storage_metadata["meta__priority"] == 1
    assert "meta__ignored" not in storage_metadata
    assert chroma_impl._build_where_clause(namespace="ns", metadata_filters={}) == {"namespace": "ns"}
    assert chroma_impl._build_where_clause(
        namespace="ns",
        metadata_filters={"kind": "decision", "ignored": ["list"]},
    ) == {"$and": [{"namespace": "ns"}, {"meta__kind": "decision"}]}

    monkeypatch.setattr(chroma_impl, "import_module", lambda _name: (_ for _ in ()).throw(ModuleNotFoundError()))
    with pytest.raises(ImportError, match="chromadb"):
        ChromaMemoryStore(client=None, persist_directory=tmp_path / "missing")

    store = ChromaMemoryStore(client=_FakeChromaClient(), persist_directory=tmp_path / "chroma")
    assert store.persist_directory == tmp_path / "chroma"
    assert store.write([], namespace="default") == []
    assert store._fetch_records_by_ids([]) == []
    assert store._records_from_get("bad-payload") == []
    assert store._records_from_get({"ids": [], "documents": "bad", "metadatas": []}) == []
    assert chroma_impl._records_from_query_results("bad-payload") == []
    assert chroma_impl._records_from_query_results({"ids": "bad", "documents": [], "metadatas": []}) == []
    assert chroma_impl._records_from_query_results({"ids": [], "documents": [], "metadatas": [], "distances": []}) == []
    store.close()


def test_chroma_memory_store_handles_short_embeddings_and_min_score_filter() -> None:
    store = ChromaMemoryStore(
        client=_FakeChromaClient(),
        embedding_provider=_ShortEmbeddingProvider(),
    )
    written = store.write(
        [
            MemoryWriteRecord(content="alpha record"),
            MemoryWriteRecord(content="beta record"),
        ],
        namespace="default",
    )
    filtered = store.search(
        MemorySearchQuery(
            text="alpha",
            namespace="default",
            top_k=5,
            min_score=0.95,
        )
    )
    store.close()

    assert len(written) == 2
    assert filtered == []


def _metadata_matches(metadata: object, where: Mapping[str, object]) -> bool:
    """Apply a minimal subset of Chroma ``where`` filtering semantics."""
    if not isinstance(metadata, Mapping):
        return False
    if "$and" in where:
        clauses = where.get("$and")
        if not isinstance(clauses, list):
            return False
        return all(_metadata_matches(metadata, clause) for clause in clauses if isinstance(clause, Mapping))
    return all(metadata.get(key) == value for key, value in where.items())


def _euclidean_distance(vector_a: list[float], vector_b: list[float]) -> float:
    """Return Euclidean distance between two vectors."""
    total = 0.0
    for value_a, value_b in zip(vector_a, vector_b, strict=True):
        delta = float(value_a) - float(value_b)
        total += delta * delta
    return math.sqrt(total)
