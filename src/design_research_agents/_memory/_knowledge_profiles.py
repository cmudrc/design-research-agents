"""Built-in engineering knowledge profile loaders and seed helpers."""

from __future__ import annotations

from collections.abc import Sequence

from design_research_agents._contracts._memory import GraphMemoryStore, MemoryStore
from design_research_agents._memory._builtin_profiles import BUILTIN_KNOWLEDGE_PROFILES
from design_research_agents._memory._knowledge_profile_types import (
    KnowledgeProfile,
    KnowledgeProfileSeedResult,
)


def list_builtin_knowledge_profiles() -> tuple[str, ...]:
    """Return available built-in knowledge profile names."""
    return tuple(sorted(BUILTIN_KNOWLEDGE_PROFILES.keys()))


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
    profile = BUILTIN_KNOWLEDGE_PROFILES.get(normalized_name)
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
    return tuple(BUILTIN_KNOWLEDGE_PROFILES[name] for name in list_builtin_knowledge_profiles())


__all__ = [
    "KnowledgeProfile",
    "KnowledgeProfileSeedResult",
    "iter_builtin_knowledge_profiles",
    "list_builtin_knowledge_profiles",
    "load_builtin_knowledge_profile",
    "seed_builtin_knowledge_profile",
]
