"""Heuristic graph extraction helpers for design relationship text."""

from __future__ import annotations

import re

from design_research_agents._contracts._memory import GraphEdgeRecord, GraphNodeRecord

_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+is connected to\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "connected_to",
    ),
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+connects to\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "connected_to",
    ),
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+drives\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "drives",
    ),
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+powers\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "powers",
    ),
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+supports\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "supports",
    ),
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+uses\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "uses",
    ),
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+depends on\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "depends_on",
    ),
    (
        re.compile(
            r"(?P<left>[A-Za-z][A-Za-z0-9 _'/-]{0,80})\s+defines\s+(?P<right>[A-Za-z][A-Za-z0-9 _'/-]{0,80})",
            re.IGNORECASE,
        ),
        "defines",
    ),
)


def extract_graph_records_from_text(text: str) -> tuple[list[GraphNodeRecord], list[GraphEdgeRecord]]:
    """Extract nodes and relationships from simple design statements.

    The extraction is intentionally heuristic and deterministic. It is useful
    for bootstrapping graph memory from structured requirement text, but it is
    not intended to replace model-based information extraction.

    Args:
        text: Source text containing simple relationship statements.

    Returns:
        Tuple ``(nodes, edges)`` extracted from the text.
    """
    nodes_by_id: dict[str, GraphNodeRecord] = {}
    edges: list[GraphEdgeRecord] = []

    normalized_sentences = _split_sentences(text)
    for sentence in normalized_sentences:
        for pattern, relationship in _RELATION_PATTERNS:
            match = pattern.search(sentence)
            if match is None:
                continue
            left_name = _normalize_entity_name(match.group("left"))
            right_name = _normalize_entity_name(match.group("right"))
            if not left_name or not right_name or left_name == right_name:
                continue

            left_node = _build_node(left_name)
            right_node = _build_node(right_name)
            nodes_by_id[left_node.node_id] = left_node
            nodes_by_id[right_node.node_id] = right_node
            edges.append(
                GraphEdgeRecord(
                    source_id=left_node.node_id,
                    target_id=right_node.node_id,
                    relationship=relationship,
                    metadata={"source_sentence": sentence},
                )
            )
            break

    return list(nodes_by_id.values()), edges


def _split_sentences(text: str) -> list[str]:
    """Split free text into small sentence-like chunks."""
    segments = re.split(r"[.;\n]+", text)
    return [segment.strip() for segment in segments if segment.strip()]


def _normalize_entity_name(value: str) -> str:
    """Normalize one extracted entity name."""
    collapsed = " ".join(value.strip().split())
    if not collapsed:
        return ""
    return collapsed


def _build_node(name: str) -> GraphNodeRecord:
    """Return one graph node for a normalized entity name."""
    return GraphNodeRecord(
        node_id=_slugify(name),
        name=name,
        node_type="component",
    )


def _slugify(value: str) -> str:
    """Return a deterministic identifier derived from ``value``."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "entity"


__all__ = ["extract_graph_records_from_text"]
