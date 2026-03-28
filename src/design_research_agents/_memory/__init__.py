"""Memory infrastructure exports."""

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphMemoryStore,
    GraphNodeRecord,
    GraphSearchQuery,
    GraphSubgraphResult,
)

from ._embedding import EmbeddingProvider, LLMEmbeddingProvider
from ._graph_extraction import extract_graph_records_from_text
from ._knowledge_ingestion import ingest_knowledge_documents
from ._knowledge_profiles import (
    KnowledgeDocument,
    KnowledgeProfile,
    KnowledgeProfileSeedResult,
    KnowledgeSource,
    iter_builtin_knowledge_profiles,
    list_builtin_knowledge_profiles,
    load_builtin_knowledge_profile,
    seed_builtin_knowledge_profile,
)
from ._stores import ChromaMemoryStore, NetworkXGraphMemoryStore, SQLiteMemoryStore

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
