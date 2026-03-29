"""Stable public memory facade exports."""

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphMemoryStore,
    GraphNodeRecord,
    GraphSearchQuery,
    GraphSubgraphResult,
)
from design_research_agents._memory._embedding import EmbeddingProvider, LLMEmbeddingProvider
from design_research_agents._memory._graph_extraction import extract_graph_records_from_text
from design_research_agents._memory._knowledge_ingestion import ingest_knowledge_documents
from design_research_agents._memory._knowledge_profiles import (
    KnowledgeDocument,
    KnowledgeProfile,
    KnowledgeProfileSeedResult,
    KnowledgeSource,
    iter_builtin_knowledge_profiles,
    list_builtin_knowledge_profiles,
    load_builtin_knowledge_profile,
    seed_builtin_knowledge_profile,
)
from design_research_agents._memory._stores._chroma_store import ChromaMemoryStore
from design_research_agents._memory._stores._networkx_graph_store import NetworkXGraphMemoryStore
from design_research_agents._memory._stores._sqlite_store import SQLiteMemoryStore

__all__ = [
    "ChromaMemoryStore",
    "EmbeddingProvider",
    "GraphEdgeRecord",
    "GraphMemoryStore",
    "GraphNodeRecord",
    "GraphSearchQuery",
    "GraphSubgraphResult",
    "KnowledgeDocument",
    "KnowledgeProfile",
    "KnowledgeProfileSeedResult",
    "KnowledgeSource",
    "LLMEmbeddingProvider",
    "NetworkXGraphMemoryStore",
    "SQLiteMemoryStore",
    "extract_graph_records_from_text",
    "ingest_knowledge_documents",
    "iter_builtin_knowledge_profiles",
    "list_builtin_knowledge_profiles",
    "load_builtin_knowledge_profile",
    "seed_builtin_knowledge_profile",
]
