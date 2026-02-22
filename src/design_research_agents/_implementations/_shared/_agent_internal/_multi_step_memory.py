"""Memory helpers for multi-step agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents._contracts._memory import (
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
)


def build_retrieval_query(
    *,
    task_prompt: str,
    memory: Sequence[Mapping[str, object]],
    memory_tail_items: int = 6,
) -> str:
    """Build retrieval query text from task prompt and recent local memory.

    Args:
        task_prompt: Current task prompt.
        memory: Local step memory entries.
        memory_tail_items: Number of trailing local memory items to include.

    Returns:
        Retrieval query text passed to the memory search layer.
    """
    memory_tail = list(memory)[-memory_tail_items:]
    return "\n".join(
        [
            f"Task: {task_prompt}",
            "Recent local memory:",
            json.dumps(memory_tail, ensure_ascii=True, sort_keys=True),
        ]
    )


def retrieve_memory_context(
    *,
    memory_store: MemoryStore | None,
    namespace: str,
    top_k: int,
    task_prompt: str,
    memory: Sequence[Mapping[str, object]],
    memory_tail_items: int = 6,
) -> tuple[str, list[dict[str, object]], str | None]:
    """Retrieve context records and return rendered context block.

    Args:
        memory_store: Optional memory store dependency.
        namespace: Namespace used for retrieval.
        top_k: Maximum number of memory matches to retrieve.
        task_prompt: Current task prompt.
        memory: Local step memory entries.
        memory_tail_items: Number of trailing local memory items for query context.

    Returns:
        Tuple ``(rendered_context, matches, error_message)``.
    """
    if memory_store is None:
        return "(none)", [], None

    query_text = build_retrieval_query(
        task_prompt=task_prompt,
        memory=memory,
        memory_tail_items=memory_tail_items,
    )
    try:
        matches = memory_store.search(
            MemorySearchQuery(
                text=query_text,
                namespace=namespace,
                top_k=max(1, int(top_k)),
            )
        )
    except Exception as exc:
        return "(none)", [], str(exc)

    serialized_matches = [match.to_dict() for match in matches]
    if not serialized_matches:
        return "(none)", [], None

    rendered_context = json.dumps(serialized_matches, ensure_ascii=True, sort_keys=True)
    return rendered_context, serialized_matches, None


def write_memory_observation(
    *,
    memory_store: MemoryStore | None,
    namespace: str,
    payload: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> str | None:
    """Write one normalized observation payload to memory.

    Args:
        memory_store: Optional memory store dependency.
        namespace: Namespace used for persistence.
        payload: Normalized observation payload to persist.
        metadata: Optional metadata associated with the payload.

    Returns:
        ``None`` on success, otherwise an error message.
    """
    if memory_store is None:
        return None

    try:
        memory_store.write(
            [
                MemoryWriteRecord(
                    content=json.dumps(payload, ensure_ascii=True, sort_keys=True),
                    metadata=dict(metadata or {}),
                )
            ],
            namespace=namespace,
        )
    except Exception as exc:
        return str(exc)
    return None


__all__ = [
    "build_retrieval_query",
    "retrieve_memory_context",
    "write_memory_observation",
]
