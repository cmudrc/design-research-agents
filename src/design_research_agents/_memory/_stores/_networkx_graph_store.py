"""NetworkX-backed graph memory store."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from types import TracebackType
from typing import Any, Self, cast

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphMemoryStore,
    GraphNodeRecord,
    GraphSearchQuery,
    GraphSubgraphResult,
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


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


def _derive_edge_id(edge: GraphEdgeRecord) -> str:
    """Return one stable edge id."""
    if edge.edge_id is not None and edge.edge_id.strip():
        return edge.edge_id.strip()
    return f"{edge.source_id}:{edge.relationship}:{edge.target_id}"


class NetworkXGraphMemoryStore(GraphMemoryStore):
    """Lightweight graph-memory backend built on top of NetworkX."""

    def __init__(self, *, networkx_module: Any | None = None) -> None:
        """Initialize the graph store.

        Args:
            networkx_module: Optional injected ``networkx`` module used for
                testing or custom graph implementations.

        Raises:
            ImportError: If ``networkx`` is unavailable.
        """
        if networkx_module is None:
            try:
                networkx_module = import_module("networkx")
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "NetworkXGraphMemoryStore requires the optional 'networkx' dependency. "
                    'Install it with `pip install -e ".[memory_graph]"`.'
                ) from exc
        self._networkx = networkx_module
        self._graphs_by_namespace: dict[str, Any] = {}

    def close(self) -> None:
        """Release any store-owned resources."""
        self._graphs_by_namespace.clear()

    def __enter__(self) -> Self:
        """Return this store for ``with``-statement usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the store on ``with``-statement exit."""
        del exc_type, exc, tb
        self.close()
        return None

    def upsert_nodes(
        self,
        nodes: Sequence[GraphNodeRecord],
        *,
        namespace: str = "default",
    ) -> list[GraphNodeRecord]:
        """Persist graph nodes and return normalized stored nodes."""
        normalized_namespace = _normalize_namespace(namespace)
        graph = self._graph_for_namespace(normalized_namespace)
        if not nodes:
            return []

        timestamp = _utc_now_iso()
        stored_nodes: list[GraphNodeRecord] = []
        for node in nodes:
            existing = graph.nodes.get(node.node_id, {})
            created_at = str(existing.get("created_at", node.created_at or timestamp))
            graph.add_node(
                node.node_id,
                name=node.name,
                node_type=node.node_type,
                description=node.description,
                metadata=dict(node.metadata),
                created_at=created_at,
                updated_at=timestamp,
            )
            stored_nodes.append(
                GraphNodeRecord(
                    node_id=node.node_id,
                    name=node.name,
                    node_type=node.node_type,
                    description=node.description,
                    metadata=dict(node.metadata),
                    created_at=created_at,
                    updated_at=timestamp,
                )
            )
        return stored_nodes

    def upsert_edges(
        self,
        edges: Sequence[GraphEdgeRecord],
        *,
        namespace: str = "default",
    ) -> list[GraphEdgeRecord]:
        """Persist graph edges and return normalized stored edges."""
        normalized_namespace = _normalize_namespace(namespace)
        graph = self._graph_for_namespace(normalized_namespace)
        if not edges:
            return []

        timestamp = _utc_now_iso()
        stored_edges: list[GraphEdgeRecord] = []
        for edge in edges:
            edge_id = _derive_edge_id(edge)
            if not graph.has_node(edge.source_id):
                graph.add_node(
                    edge.source_id,
                    name=edge.source_id,
                    node_type="entity",
                    description="",
                    metadata={},
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            if not graph.has_node(edge.target_id):
                graph.add_node(
                    edge.target_id,
                    name=edge.target_id,
                    node_type="entity",
                    description="",
                    metadata={},
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            existing = graph.get_edge_data(edge.source_id, edge.target_id, key=edge_id, default={})
            created_at = str(existing.get("created_at", edge.created_at or timestamp))
            graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge_id,
                relationship=edge.relationship,
                metadata=dict(edge.metadata),
                created_at=created_at,
                updated_at=timestamp,
            )
            stored_edges.append(
                GraphEdgeRecord(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relationship=edge.relationship,
                    edge_id=edge_id,
                    metadata=dict(edge.metadata),
                    created_at=created_at,
                    updated_at=timestamp,
                )
            )
        return stored_edges

    def query_subgraph(self, query: GraphSearchQuery) -> GraphSubgraphResult:
        """Retrieve one relevant graph subgraph for a structured query."""
        normalized_namespace = _normalize_namespace(query.namespace)
        graph = self._graphs_by_namespace.get(normalized_namespace)
        if graph is None:
            return GraphSubgraphResult(
                namespace=normalized_namespace,
                query_text=query.text,
            )

        top_k = max(1, int(query.top_k))
        max_hops = max(0, int(query.max_hops))
        query_tokens = _tokenize(str(query.text))
        allowed_node_types = {value for value in query.node_type_filters if value}

        candidate_nodes: list[GraphNodeRecord] = []
        for node_id, attrs in graph.nodes(data=True):
            node_type = str(attrs.get("node_type", "entity"))
            if allowed_node_types and node_type not in allowed_node_types:
                continue

            metadata = attrs.get("metadata", {})
            metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
            if not _metadata_matches(metadata=metadata_dict, filters=query.metadata_filters):
                continue

            searchable_text = " ".join(
                [
                    str(attrs.get("name", "")),
                    str(attrs.get("description", "")),
                    " ".join(f"{key} {value}" for key, value in sorted(metadata_dict.items())),
                ]
            )
            score = _lexical_score(query_tokens=query_tokens, content=searchable_text)
            if query.min_score is not None and score < float(query.min_score):
                continue

            candidate_nodes.append(
                GraphNodeRecord(
                    node_id=str(node_id),
                    name=str(attrs.get("name", node_id)),
                    node_type=node_type,
                    description=str(attrs.get("description", "")),
                    metadata=metadata_dict,
                    created_at=_optional_string(attrs.get("created_at")),
                    updated_at=_optional_string(attrs.get("updated_at")),
                    score=round(score, 8),
                )
            )

        candidate_nodes.sort(
            key=lambda node: (
                float(node.score or 0.0),
                node.name,
                node.node_id,
            ),
            reverse=True,
        )
        matched_nodes = candidate_nodes[:top_k]
        if not matched_nodes:
            return GraphSubgraphResult(
                namespace=normalized_namespace,
                query_text=query.text,
            )

        included_node_ids = self._expand_node_ids(
            graph=graph,
            seed_node_ids=[node.node_id for node in matched_nodes],
            max_hops=max_hops,
        )
        nodes_by_id = {node.node_id: node for node in matched_nodes}
        for node_id in included_node_ids:
            if node_id in nodes_by_id:
                continue
            attrs = graph.nodes[node_id]
            metadata = attrs.get("metadata", {})
            metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
            nodes_by_id[node_id] = GraphNodeRecord(
                node_id=str(node_id),
                name=str(attrs.get("name", node_id)),
                node_type=str(attrs.get("node_type", "entity")),
                description=str(attrs.get("description", "")),
                metadata=metadata_dict,
                created_at=_optional_string(attrs.get("created_at")),
                updated_at=_optional_string(attrs.get("updated_at")),
            )

        relationship_filters = {value for value in query.relationship_filters if value}
        edges: list[GraphEdgeRecord] = []
        for source_id, target_id, edge_key, attrs in graph.edges(keys=True, data=True):
            if source_id not in included_node_ids or target_id not in included_node_ids:
                continue
            relationship = str(attrs.get("relationship", "related_to"))
            if relationship_filters and relationship not in relationship_filters:
                continue
            metadata = attrs.get("metadata", {})
            metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
            edges.append(
                GraphEdgeRecord(
                    source_id=str(source_id),
                    target_id=str(target_id),
                    relationship=relationship,
                    edge_id=str(edge_key),
                    metadata=metadata_dict,
                    created_at=_optional_string(attrs.get("created_at")),
                    updated_at=_optional_string(attrs.get("updated_at")),
                )
            )
        edges.sort(key=lambda edge: (edge.source_id, edge.target_id, edge.relationship, edge.edge_id or ""))

        ordered_nodes = sorted(
            nodes_by_id.values(),
            key=lambda node: (
                float(node.score or -1.0),
                node.name,
                node.node_id,
            ),
            reverse=True,
        )
        return GraphSubgraphResult(
            namespace=normalized_namespace,
            query_text=query.text,
            matched_node_ids=tuple(node.node_id for node in matched_nodes),
            nodes=tuple(ordered_nodes),
            edges=tuple(edges),
        )

    def _graph_for_namespace(self, namespace: str) -> Any:
        """Return the mutable graph for one namespace, creating it if needed."""
        graph = self._graphs_by_namespace.get(namespace)
        if graph is None:
            graph = cast(Any, self._networkx).MultiDiGraph()
            self._graphs_by_namespace[namespace] = graph
        return graph

    def _expand_node_ids(
        self,
        *,
        graph: Any,
        seed_node_ids: Sequence[str],
        max_hops: int,
    ) -> set[str]:
        """Return node ids reachable from seed nodes within ``max_hops``."""
        visited = set(seed_node_ids)
        if max_hops == 0:
            return visited

        queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in seed_node_ids)
        while queue:
            current_node_id, depth = queue.popleft()
            if depth >= max_hops:
                continue
            neighbor_ids = set(graph.successors(current_node_id)).union(set(graph.predecessors(current_node_id)))
            for neighbor_id in neighbor_ids:
                if neighbor_id in visited:
                    continue
                visited.add(str(neighbor_id))
                queue.append((str(neighbor_id), depth + 1))
        return visited


def _optional_string(value: object) -> str | None:
    """Return a trimmed string when ``value`` is string-like, otherwise ``None``."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = ["NetworkXGraphMemoryStore"]
