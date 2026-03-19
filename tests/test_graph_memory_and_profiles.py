"""Graph memory and built-in knowledge profile tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphMemoryStore,
    GraphNodeRecord,
    GraphSearchQuery,
    GraphSubgraphResult,
    MemorySearchQuery,
)
from design_research_agents._memory import _graph_extraction as graph_extraction_impl
from design_research_agents._memory._stores import _networkx_graph_store as graph_store_impl
from design_research_agents.memory import (
    NetworkXGraphMemoryStore,
    SQLiteMemoryStore,
    extract_graph_records_from_text,
    iter_builtin_knowledge_profiles,
    list_builtin_knowledge_profiles,
    load_builtin_knowledge_profile,
    seed_builtin_knowledge_profile,
)


class _FakeNodeView:
    """Tiny callable mapping that emulates ``networkx`` node views."""

    def __init__(self, storage: dict[str, dict[str, object]]) -> None:
        self._storage = storage

    def __call__(self, data: bool = False) -> list[object]:
        if data:
            return list(self._storage.items())
        return list(self._storage.keys())

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._storage.get(key, default)

    def __getitem__(self, key: str) -> dict[str, object]:
        return self._storage[key]


class _FakeEdgeView:
    """Tiny callable mapping that emulates ``networkx`` edge views."""

    def __init__(self, storage: dict[tuple[str, str, str], dict[str, object]]) -> None:
        self._storage = storage

    def __call__(self, *, keys: bool = False, data: bool = False) -> list[object]:
        if not keys or not data:
            raise AssertionError("Test fake only supports edges(keys=True, data=True).")
        return [
            (source_id, target_id, edge_id, attrs) for (source_id, target_id, edge_id), attrs in self._storage.items()
        ]


class _FakeMultiDiGraph:
    """Small subset of ``networkx.MultiDiGraph`` needed by the store tests."""

    def __init__(self) -> None:
        self._node_storage: dict[str, dict[str, object]] = {}
        self._edge_storage: dict[tuple[str, str, str], dict[str, object]] = {}
        self.nodes = _FakeNodeView(self._node_storage)
        self.edges = _FakeEdgeView(self._edge_storage)

    def add_node(self, node_id: str, **attrs: object) -> None:
        self._node_storage[node_id] = dict(attrs)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._node_storage

    def add_edge(self, source_id: str, target_id: str, key: str, **attrs: object) -> None:
        self._edge_storage[(source_id, target_id, key)] = dict(attrs)

    def get_edge_data(self, source_id: str, target_id: str, key: str, default: object | None = None) -> object | None:
        return self._edge_storage.get((source_id, target_id, key), default)

    def successors(self, node_id: str) -> list[str]:
        return [target_id for source_id, target_id, _edge_id in self._edge_storage if source_id == node_id]

    def predecessors(self, node_id: str) -> list[str]:
        return [source_id for source_id, target_id, _edge_id in self._edge_storage if target_id == node_id]


class _FakeNetworkXModule:
    """Namespace wrapper exposing the fake graph class like the real module."""

    MultiDiGraph = _FakeMultiDiGraph


class _ProtocolGraphStore(GraphMemoryStore):
    """Concrete graph store used to exercise protocol default methods."""

    def upsert_nodes(self, nodes, *, namespace: str = "default"):
        del namespace
        return list(nodes)

    def upsert_edges(self, edges, *, namespace: str = "default"):
        del namespace
        return list(edges)

    def query_subgraph(self, query):
        return GraphSubgraphResult(namespace=query.namespace, query_text=query.text)


def test_networkx_graph_memory_store_supports_nodes_edges_and_traversal() -> None:
    store = NetworkXGraphMemoryStore(networkx_module=_FakeNetworkXModule())
    store.upsert_nodes(
        [
            GraphNodeRecord(node_id="motor-a", name="Motor A", node_type="component"),
            GraphNodeRecord(node_id="gearbox-b", name="Gearbox B", node_type="component"),
            GraphNodeRecord(node_id="shaft-c", name="Shaft C", node_type="component"),
        ],
        namespace="design",
    )
    store.upsert_edges(
        [
            GraphEdgeRecord(source_id="motor-a", target_id="gearbox-b", relationship="drives"),
            GraphEdgeRecord(source_id="gearbox-b", target_id="shaft-c", relationship="supports"),
        ],
        namespace="design",
    )

    result = store.query_subgraph(
        GraphSearchQuery(
            text="How does Motor A connect to Gearbox B?",
            namespace="design",
            top_k=2,
            max_hops=1,
        )
    )
    store.close()

    assert result.namespace == "design"
    assert "motor-a" in result.matched_node_ids
    assert any(node.node_id == "gearbox-b" for node in result.nodes)
    assert any(edge.relationship == "drives" for edge in result.edges)


def test_extract_graph_records_from_text_produces_nodes_and_edges() -> None:
    nodes, edges = extract_graph_records_from_text("Motor A drives Gearbox B. Gearbox B supports Shaft C.")

    assert {node.node_id for node in nodes} == {"motor-a", "gearbox-b", "shaft-c"}
    assert [edge.relationship for edge in edges] == ["drives", "supports"]


def test_graph_contract_dataclasses_and_protocol_defaults() -> None:
    node = GraphNodeRecord(node_id="spring", name="Spring", node_type="component")
    edge = GraphEdgeRecord(source_id="spring", target_id="bracket", relationship="supports")
    subgraph = GraphSubgraphResult(
        namespace="design",
        query_text="spring support",
        matched_node_ids=("spring",),
        nodes=(node,),
        edges=(edge,),
    )
    store = _ProtocolGraphStore()

    assert node.to_dict()["node_id"] == "spring"
    assert edge.to_dict()["relationship"] == "supports"
    assert subgraph.to_dict()["matched_node_ids"] == ["spring"]
    assert store.__enter__() is store
    assert store.close() is None
    assert store.__exit__(None, None, None) is None


def test_graph_extraction_and_store_helper_branches(monkeypatch) -> None:
    nodes, edges = extract_graph_records_from_text("Motor A drives Motor A.   ")
    assert nodes == []
    assert edges == []
    assert graph_extraction_impl._normalize_entity_name("   ") == ""
    assert graph_extraction_impl._slugify("!!!") == "entity"

    assert graph_store_impl._lexical_score(query_tokens=[], content="alpha") == 0.0
    assert graph_store_impl._lexical_score(query_tokens=["alpha"], content="") == 0.0
    assert (
        graph_store_impl._derive_edge_id(
            GraphEdgeRecord(source_id="a", target_id="b", relationship="links", edge_id="edge-1")
        )
        == "edge-1"
    )
    assert graph_store_impl._optional_string(None) is None

    monkeypatch.setattr(graph_store_impl, "import_module", lambda _name: (_ for _ in ()).throw(ModuleNotFoundError()))
    with pytest.raises(ImportError, match="networkx"):
        NetworkXGraphMemoryStore(networkx_module=None)

    with NetworkXGraphMemoryStore(networkx_module=_FakeNetworkXModule()) as store:
        assert store is not None
        assert store.upsert_nodes([], namespace="design") == []
        assert store.upsert_edges([], namespace="design") == []
        auto_edges = store.upsert_edges(
            [GraphEdgeRecord(source_id="auto-a", target_id="auto-b", relationship="links")],
            namespace="design",
        )
        missing_result = store.query_subgraph(GraphSearchQuery(text="missing", namespace="missing"))
    assert auto_edges[0].source_id == "auto-a"
    assert auto_edges[0].target_id == "auto-b"
    assert missing_result.nodes == ()


def test_networkx_graph_memory_store_filter_and_zero_hop_branches() -> None:
    store = NetworkXGraphMemoryStore(networkx_module=_FakeNetworkXModule())
    store.upsert_nodes(
        [
            GraphNodeRecord(node_id="motor-a", name="Motor A", node_type="component", metadata={"kind": "driver"}),
            GraphNodeRecord(node_id="gearbox-b", name="Gearbox B", node_type="component", metadata={"kind": "driven"}),
        ],
        namespace="design",
    )
    store.upsert_edges(
        [GraphEdgeRecord(source_id="motor-a", target_id="gearbox-b", relationship="drives")],
        namespace="design",
    )

    no_match = store.query_subgraph(
        GraphSearchQuery(
            text="motor",
            namespace="design",
            top_k=1,
            node_type_filters=("formula",),
        )
    )
    zero_hop = store.query_subgraph(
        GraphSearchQuery(
            text="motor",
            namespace="design",
            top_k=1,
            max_hops=0,
            relationship_filters=("supports",),
            metadata_filters={"kind": "driver"},
            min_score=0.1,
        )
    )
    store.close()

    assert no_match.nodes == ()
    assert len(zero_hop.nodes) == 1
    assert zero_hop.edges == ()


def test_built_in_mechanics_profile_can_seed_text_and_graph_stores(tmp_path: Path) -> None:
    memory_store = SQLiteMemoryStore(db_path=tmp_path / "mechanics.sqlite3")
    graph_store = NetworkXGraphMemoryStore(networkx_module=_FakeNetworkXModule())

    seed_result = seed_builtin_knowledge_profile(
        "mechanics",
        memory_store=memory_store,
        graph_store=graph_store,
        namespace="mechanics",
    )

    matches = memory_store.search(
        MemorySearchQuery(
            text="Hooke's law spring constant",
            namespace="mechanics",
            top_k=3,
        )
    )
    subgraph = graph_store.query_subgraph(
        GraphSearchQuery(
            text="beam bending moment",
            namespace="mechanics",
            top_k=2,
            max_hops=1,
        )
    )
    memory_store.close()
    graph_store.close()

    assert seed_result.memory_records_written >= 4
    assert seed_result.graph_nodes_written >= 5
    assert matches
    assert matches[0].metadata["profile"] == "mechanics"
    assert "Hooke's law" in matches[0].content
    assert any(node.node_id == "beam-bending-stress" for node in subgraph.nodes)
    assert any(edge.relationship == "depends_on" for edge in subgraph.edges)


def test_built_in_profile_helpers_validate_names(tmp_path: Path) -> None:
    assert "mechanics" in list_builtin_knowledge_profiles()
    assert load_builtin_knowledge_profile("aerospace").name == "aerospace"
    assert load_builtin_knowledge_profile("stem").to_dict()["name"] == "stem"
    assert [profile.name for profile in iter_builtin_knowledge_profiles()] == ["aerospace", "mechanics", "stem"]

    with pytest.raises(ValueError, match="Unknown knowledge profile"):
        load_builtin_knowledge_profile("unknown-domain")

    with pytest.raises(ValueError, match="At least one of memory_store or graph_store"):
        seed_builtin_knowledge_profile("stem", memory_store=None, graph_store=None)

    with SQLiteMemoryStore(db_path=tmp_path / "stem.sqlite3") as store:
        seed_result = seed_builtin_knowledge_profile("stem", memory_store=store)
    assert seed_result.to_dict()["profile_name"] == "stem"
