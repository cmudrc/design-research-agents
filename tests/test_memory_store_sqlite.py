"""SQLite memory store contract tests."""

from __future__ import annotations

from pathlib import Path

from design_research_agents.contracts.memory import MemorySearchQuery, MemoryWriteRecord
from design_research_agents.memory import EmbeddingProvider, SQLiteMemoryStore


class _StaticEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for ranking tests."""

    def __init__(self, *, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    @property
    def model_name(self) -> str:
        return "static-test-embeddings"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]] | None:
        return [list(self._vectors_by_text.get(text, [0.0, 0.0])) for text in texts]


class _FallbackEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that always falls back to lexical search."""

    @property
    def model_name(self) -> str:
        return "fallback-model"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]] | None:
        del texts
        return None


def test_sqlite_memory_store_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"

    first_store = SQLiteMemoryStore(db_path=db_path)
    first_store.write(
        [
            MemoryWriteRecord(
                content="alpha memory artifact",
                metadata={"project": "one"},
            )
        ],
        namespace="ns",
    )
    first_store.close()

    second_store = SQLiteMemoryStore(db_path=db_path)
    matches = second_store.search(MemorySearchQuery(text="alpha artifact", namespace="ns", top_k=3))
    second_store.close()

    assert len(matches) == 1
    assert matches[0].content == "alpha memory artifact"
    assert matches[0].namespace == "ns"
    assert matches[0].metadata["project"] == "one"


def test_sqlite_memory_store_respects_namespace_isolation(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write([MemoryWriteRecord(content="shared phrase")], namespace="alpha")
    store.write([MemoryWriteRecord(content="shared phrase")], namespace="beta")

    alpha_matches = store.search(MemorySearchQuery(text="shared", namespace="alpha", top_k=5))
    beta_matches = store.search(MemorySearchQuery(text="shared", namespace="beta", top_k=5))

    store.close()

    assert len(alpha_matches) == 1
    assert len(beta_matches) == 1
    assert alpha_matches[0].namespace == "alpha"
    assert beta_matches[0].namespace == "beta"


def test_sqlite_memory_store_applies_metadata_filters(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [
            MemoryWriteRecord(content="design note", metadata={"kind": "note", "phase": 1}),
            MemoryWriteRecord(content="design decision", metadata={"kind": "decision", "phase": 1}),
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


def test_sqlite_memory_store_lexical_ranking_orders_more_relevant_first(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [
            MemoryWriteRecord(content="alpha beta gamma"),
            MemoryWriteRecord(content="alpha"),
            MemoryWriteRecord(content="delta epsilon"),
        ],
        namespace="default",
    )

    matches = store.search(MemorySearchQuery(text="alpha beta", top_k=3, namespace="default"))
    store.close()

    assert len(matches) == 3
    assert matches[0].content == "alpha beta gamma"
    assert matches[0].lexical_score is not None
    assert matches[0].score is not None
    assert matches[0].score >= matches[1].score


def test_sqlite_memory_store_uses_vector_weighting_when_embeddings_present(tmp_path: Path) -> None:
    provider = _StaticEmbeddingProvider(
        vectors_by_text={
            "query": [1.0, 0.0],
            "mostly lexical": [0.0, 1.0],
            "mostly vector": [1.0, 0.0],
        }
    )
    store = SQLiteMemoryStore(
        db_path=tmp_path / "memory.sqlite3",
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

    assert len(matches) == 2
    assert matches[0].content == "mostly vector"
    assert matches[0].vector_score is not None


def test_sqlite_memory_store_falls_back_to_lexical_when_embedding_provider_returns_none(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryStore(
        db_path=tmp_path / "memory.sqlite3",
        embedding_provider=_FallbackEmbeddingProvider(),
    )
    store.write([MemoryWriteRecord(content="fallback lexical retrieval")], namespace="default")

    matches = store.search(MemorySearchQuery(text="fallback", namespace="default", top_k=1))
    store.close()

    assert len(matches) == 1
    assert matches[0].vector_score is None
    assert matches[0].lexical_score is not None
