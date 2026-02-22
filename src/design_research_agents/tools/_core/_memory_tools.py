"""Memory-oriented core tools for retrieval and write-back."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from design_research_agents._contracts._memory import (
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
)
from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents._memory import SQLiteMemoryStore
from design_research_agents.tools._policy import ToolPolicy
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str


@dataclass(slots=True, frozen=True)
class _ResolvedStore:
    """Resolved store binding for one memory tool call."""

    store: MemoryStore
    close_after: bool
    db_path: Path


def register_memory_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register SQLite-first memory tools.

    Args:
        source: In-process tool source receiving registrations.
        policy: Runtime policy used for path validation.
    """
    read_metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_read=True),
        timeout_s=15,
        max_output_bytes=131_072,
        risky=False,
    )
    write_metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_read=True, filesystem_write=True),
        timeout_s=20,
        max_output_bytes=131_072,
        risky=True,
    )

    source.register_tool(
        spec=ToolSpec(
            name="memory.search",
            description=("Search persistent memory records by lexical relevance and optional vector similarity."),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "namespace": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "min_score": {"type": "number"},
                    "metadata_filters": {"type": "object"},
                    "db_path": {"type": "string"},
                },
                "required": ["text"],
            },
            output_schema={"type": "object"},
            metadata=read_metadata,
        ),
        handler=lambda i, r, d: _memory_search(i, dependencies=d, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="memory.write",
            description=("Write records to the persistent memory store for downstream retrieval and RAG."),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "content": {"type": "string"},
                                "metadata": {"type": "object"},
                                "item_id": {"type": "string"},
                            },
                            "required": ["content"],
                        },
                    },
                    "namespace": {"type": "string"},
                    "db_path": {"type": "string"},
                },
                "required": ["records"],
            },
            output_schema={"type": "object"},
            metadata=write_metadata,
        ),
        handler=lambda i, r, d: _memory_write(i, dependencies=d, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="memory.stats",
            description=(
                "Return namespace-level memory table stats (record count, embeddings, latest update timestamp)."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "namespace": {"type": "string"},
                    "db_path": {"type": "string"},
                },
            },
            output_schema={"type": "object"},
            metadata=read_metadata,
        ),
        handler=lambda i, r, d: _memory_stats(i, dependencies=d, policy=policy),
    )


def _memory_search(
    input_dict: Mapping[str, object],
    *,
    dependencies: Mapping[str, object],
    policy: ToolPolicy,
) -> Mapping[str, object]:
    """Run one memory search request.

    Args:
        input_dict: Tool input payload.
        dependencies: Runtime dependency mapping.
        policy: Runtime policy.

    Returns:
        Memory retrieval payload with normalized query and matches.
    """
    query_text = get_str(input_dict, "text").strip()
    if not query_text:
        raise ValueError("text is required.")

    namespace = _normalize_namespace(get_str(input_dict, "namespace", default="default"))
    top_k = get_int(input_dict, "top_k", default=5)
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    min_score = _extract_optional_float(input_dict.get("min_score"), key="min_score")
    metadata_filters = _extract_metadata_filters(input_dict.get("metadata_filters"))

    resolved = _resolve_store(input_dict=input_dict, dependencies=dependencies, policy=policy)
    try:
        query = MemorySearchQuery(
            text=query_text,
            namespace=namespace,
            top_k=top_k,
            min_score=min_score,
            metadata_filters=metadata_filters,
        )
        matches = resolved.store.search(query)
    finally:
        if resolved.close_after:
            _close_store(resolved.store)

    serialized_matches = [record.to_dict() for record in matches]
    retrieval_mode = "hybrid_vector" if any(match.vector_score is not None for match in matches) else "lexical"
    return {
        "query": query.to_dict(),
        "matches": serialized_matches,
        "count": len(serialized_matches),
        "retrieval_mode": retrieval_mode,
    }


def _memory_write(
    input_dict: Mapping[str, object],
    *,
    dependencies: Mapping[str, object],
    policy: ToolPolicy,
) -> Mapping[str, object]:
    """Run one memory write request.

    Args:
        input_dict: Tool input payload.
        dependencies: Runtime dependency mapping.
        policy: Runtime policy.

    Returns:
        Write summary payload.
    """
    namespace = _normalize_namespace(get_str(input_dict, "namespace", default="default"))
    records = _normalize_write_records(input_dict.get("records"))

    resolved = _resolve_store(input_dict=input_dict, dependencies=dependencies, policy=policy)
    try:
        written = resolved.store.write(records, namespace=namespace)
    finally:
        if resolved.close_after:
            _close_store(resolved.store)

    return {
        "written": len(written),
        "ids": [record.item_id for record in written],
        "namespace": namespace,
    }


def _memory_stats(
    input_dict: Mapping[str, object],
    *,
    dependencies: Mapping[str, object],
    policy: ToolPolicy,
) -> Mapping[str, object]:
    """Collect namespace-level SQLite memory statistics.

    Args:
        input_dict: Tool input payload.
        dependencies: Runtime dependency mapping.
        policy: Runtime policy.

    Returns:
        Memory stats payload.
    """
    namespace = _normalize_namespace(get_str(input_dict, "namespace", default="default"))
    db_path = _resolve_stats_db_path(
        input_dict=input_dict,
        dependencies=dependencies,
        policy=policy,
    )

    bootstrap = SQLiteMemoryStore(db_path=db_path)
    bootstrap.close()

    with sqlite3.connect(str(db_path)) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS record_count,
                MAX(updated_at) AS latest_updated_at
            FROM memory_items
            WHERE namespace = ?
            """,
            (namespace,),
        ).fetchone()
        embedding_row = connection.execute(
            """
            SELECT COUNT(*) AS embedding_count
            FROM memory_embeddings AS e
            INNER JOIN memory_items AS i
                ON i.id = e.item_id
            WHERE i.namespace = ?
            """,
            (namespace,),
        ).fetchone()

    record_count = int(row[0]) if row is not None and row[0] is not None else 0
    latest_updated_at = str(row[1]) if row is not None and row[1] is not None else None
    embedding_count = int(embedding_row[0]) if embedding_row is not None and embedding_row[0] is not None else 0
    return {
        "record_count": record_count,
        "embedding_count": embedding_count,
        "latest_updated_at": latest_updated_at,
        "namespace": namespace,
        "db_path": str(db_path),
    }


def _resolve_store(
    *,
    input_dict: Mapping[str, object],
    dependencies: Mapping[str, object],
    policy: ToolPolicy,
) -> _ResolvedStore:
    """Resolve memory store from dependencies or create SQLite fallback.

    Args:
        input_dict: Tool input payload.
        dependencies: Runtime dependency mapping.
        policy: Runtime policy.

    Returns:
        Resolved store binding.
    """
    dependency_store = dependencies.get("memory_store")
    if dependency_store is not None:
        if not _is_memory_store_like(dependency_store):
            raise ValueError("dependencies['memory_store'] must expose callable write(...) and search(...).")
        typed_dependency_store = cast(MemoryStore, dependency_store)
        dependency_db_path = None
        if hasattr(dependency_store, "db_path"):
            dependency_db_path = cast(Any, dependency_store).db_path
        if isinstance(dependency_db_path, Path):
            resolved_path = policy.resolve_write_path(str(dependency_db_path))
        elif isinstance(dependency_db_path, str) and dependency_db_path.strip():
            resolved_path = policy.resolve_write_path(dependency_db_path)
        else:
            resolved_path = _resolve_db_path(input_dict=input_dict, policy=policy)
        return _ResolvedStore(
            store=typed_dependency_store,
            close_after=False,
            db_path=resolved_path,
        )

    db_path = _resolve_db_path(input_dict=input_dict, policy=policy)
    return _ResolvedStore(
        store=SQLiteMemoryStore(db_path=db_path),
        close_after=True,
        db_path=db_path,
    )


def _resolve_stats_db_path(
    *,
    input_dict: Mapping[str, object],
    dependencies: Mapping[str, object],
    policy: ToolPolicy,
) -> Path:
    """Resolve sqlite path used by ``memory.stats``.

    Args:
        input_dict: Tool input payload.
        dependencies: Runtime dependency mapping.
        policy: Runtime policy.

    Returns:
        Resolved sqlite file path.
    """
    raw_db_path = input_dict.get("db_path")
    if isinstance(raw_db_path, str) and raw_db_path.strip():
        return policy.resolve_write_path(raw_db_path.strip())

    dependency_store = dependencies.get("memory_store")
    dependency_db_path = None
    if dependency_store is not None and hasattr(dependency_store, "db_path"):
        dependency_db_path = cast(Any, dependency_store).db_path
    if isinstance(dependency_db_path, Path):
        return policy.resolve_write_path(str(dependency_db_path))
    if isinstance(dependency_db_path, str) and dependency_db_path.strip():
        return policy.resolve_write_path(dependency_db_path.strip())

    return _resolve_db_path(input_dict=input_dict, policy=policy)


def _resolve_db_path(*, input_dict: Mapping[str, object], policy: ToolPolicy) -> Path:
    """Resolve memory sqlite path under policy rules.

    Args:
        input_dict: Tool input payload.
        policy: Runtime policy.

    Returns:
        Resolved writable sqlite path.
    """
    raw_db_path = input_dict.get("db_path")
    if isinstance(raw_db_path, str) and raw_db_path.strip():
        return policy.resolve_write_path(raw_db_path.strip())
    default_path = policy.artifacts_root / "memory" / "memory.sqlite3"
    return policy.resolve_write_path(str(default_path))


def _normalize_namespace(value: str) -> str:
    """Normalize namespace values.

    Args:
        value: Raw namespace text.

    Returns:
        Normalized namespace.
    """
    normalized = value.strip()
    return normalized if normalized else "default"


def _extract_metadata_filters(raw_filters: object) -> dict[str, object]:
    """Validate metadata filter payload.

    Args:
        raw_filters: Raw metadata_filters input.

    Returns:
        Normalized metadata filter object.
    """
    if raw_filters is None:
        return {}
    if not isinstance(raw_filters, Mapping):
        raise ValueError("metadata_filters must be an object.")
    return dict(raw_filters)


def _extract_optional_float(raw_value: object, *, key: str) -> float | None:
    """Extract one optional float value from a mixed raw input.

    Args:
        raw_value: Raw value.
        key: Field name for errors.

    Returns:
        Parsed float or ``None``.
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be a number.")
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            return float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{key} must be a number.") from exc
    raise ValueError(f"{key} must be a number.")


def _normalize_write_records(raw_records: object) -> list[MemoryWriteRecord]:
    """Normalize write payload into ``MemoryWriteRecord`` objects.

    Args:
        raw_records: Raw records payload.

    Returns:
        Normalized records.
    """
    if not isinstance(raw_records, list):
        raise ValueError("records must be a list.")

    normalized: list[MemoryWriteRecord] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"records[{index}] must be an object.")
        content = record.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"records[{index}].content must be a non-empty string.")
        raw_metadata = record.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(f"records[{index}].metadata must be an object when provided.")
        raw_item_id = record.get("item_id")
        item_id = raw_item_id.strip() if isinstance(raw_item_id, str) else None
        normalized.append(
            MemoryWriteRecord(
                content=content,
                metadata=dict(raw_metadata),
                item_id=item_id or None,
            )
        )
    return normalized


def _is_memory_store_like(candidate: object) -> bool:
    """Check protocol-compatible memory store surface.

    Args:
        candidate: Candidate runtime object.

    Returns:
        ``True`` when candidate exposes ``search`` and ``write`` callables.
    """
    if not hasattr(candidate, "search") or not hasattr(candidate, "write"):
        return False
    candidate_any = cast(Any, candidate)
    search_callable = candidate_any.search
    write_callable = candidate_any.write
    return callable(search_callable) and callable(write_callable)


def _close_store(store: object) -> None:
    """Close store objects when close hook exists.

    Args:
        store: Store object to close.
    """
    if hasattr(store, "close"):
        store.close()


__all__ = ["register_memory_tools"]
