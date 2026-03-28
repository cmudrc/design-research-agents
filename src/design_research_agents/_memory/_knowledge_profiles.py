"""Built-in engineering knowledge profile loaders and seed helpers."""

from __future__ import annotations

from collections.abc import Sequence

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphMemoryStore,
    GraphNodeRecord,
    MemoryStore,
    MemoryWriteRecord,
)
from design_research_agents._memory._knowledge_ingestion import ingest_knowledge_documents
from design_research_agents._memory._knowledge_profile_types import (
    KnowledgeDocument,
    KnowledgeProfile,
    KnowledgeProfileSeedResult,
    KnowledgeSource,
)
from design_research_agents._memory._knowledge_resource_loader import (
    list_packaged_knowledge_profile_names,
    load_packaged_knowledge_profile,
)


def list_builtin_knowledge_profiles() -> tuple[str, ...]:
    """Return available built-in knowledge profile names."""
    return list_packaged_knowledge_profile_names()


def load_builtin_knowledge_profile(profile_name: str) -> KnowledgeProfile:
    """Return one built-in knowledge profile by name.

    Args:
        profile_name: Built-in profile name.

    Returns:
        Built-in knowledge profile.

    Raises:
        ValueError: If the requested profile is unknown.
    """
    return _clone_knowledge_profile(load_packaged_knowledge_profile(profile_name))


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
    return tuple(load_builtin_knowledge_profile(name) for name in list_builtin_knowledge_profiles())


def _clone_knowledge_profile(profile: KnowledgeProfile) -> KnowledgeProfile:
    """Return a copy of one profile with isolated mutable nested metadata."""
    return KnowledgeProfile(
        name=profile.name,
        description=profile.description,
        records=tuple(
            MemoryWriteRecord(
                content=record.content,
                metadata=dict(record.metadata),
                item_id=record.item_id,
            )
            for record in profile.records
        ),
        graph_nodes=tuple(
            GraphNodeRecord(
                node_id=node.node_id,
                name=node.name,
                node_type=node.node_type,
                description=node.description,
                metadata=dict(node.metadata),
                created_at=node.created_at,
                updated_at=node.updated_at,
                score=node.score,
            )
            for node in profile.graph_nodes
        ),
        graph_edges=tuple(
            GraphEdgeRecord(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship,
                edge_id=edge.edge_id,
                metadata=dict(edge.metadata),
                created_at=edge.created_at,
                updated_at=edge.updated_at,
            )
            for edge in profile.graph_edges
        ),
        sources=tuple(
            KnowledgeSource(
                label=source.label,
                uri=source.uri,
                kind=source.kind,
            )
            for source in profile.sources
        ),
    )


__all__ = [
    "KnowledgeDocument",
    "KnowledgeProfile",
    "KnowledgeProfileSeedResult",
    "KnowledgeSource",
    "ingest_knowledge_documents",
    "iter_builtin_knowledge_profiles",
    "list_builtin_knowledge_profiles",
    "load_builtin_knowledge_profile",
    "seed_builtin_knowledge_profile",
]
