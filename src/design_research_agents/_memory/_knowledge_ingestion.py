"""Deterministic document ingestion helpers for built-in knowledge profiles."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from design_research_agents._contracts._memory import (
    GraphEdgeRecord,
    GraphNodeRecord,
    MemoryWriteRecord,
)
from design_research_agents._memory._graph_extraction import extract_graph_records_from_text
from design_research_agents._memory._knowledge_profile_types import (
    KnowledgeDocument,
    KnowledgeProfile,
    KnowledgeSource,
)

_SECTION_HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_PARAGRAPH_BREAK_PATTERN = re.compile(r"\n\s*\n+")


@dataclass(slots=True, frozen=True, kw_only=True)
class _MarkdownChunk:
    """One deterministic chunk derived from a canonical knowledge document."""

    section: str
    content: str


def ingest_knowledge_documents(
    profile_name: str,
    *,
    description: str,
    documents: Sequence[KnowledgeDocument],
    sources: Sequence[KnowledgeSource] = (),
) -> KnowledgeProfile:
    """Ingest canonical knowledge documents into one materialized profile.

    Args:
        profile_name: Stable profile name.
        description: Human-readable profile summary.
        documents: Canonical documents to ingest.
        sources: Optional structured provenance entries for the profile.

    Returns:
        Materialized knowledge profile with text records and heuristic graph data.

    Raises:
        ValueError: If required values are missing or invalid.
    """
    normalized_profile_name = profile_name.strip()
    if not normalized_profile_name:
        raise ValueError("profile_name must be non-empty.")

    normalized_description = description.strip()
    if not normalized_description:
        raise ValueError("description must be non-empty.")

    normalized_documents = tuple(_normalize_document(document) for document in documents)
    if not normalized_documents:
        raise ValueError("documents must include at least one canonical knowledge document.")

    records: list[MemoryWriteRecord] = []
    graph_nodes_by_id: dict[str, GraphNodeRecord] = {}
    graph_edges_by_id: dict[str, GraphEdgeRecord] = {}

    for document in normalized_documents:
        chunks = _chunk_markdown_document(document)
        for chunk_index, chunk in enumerate(chunks, start=1):
            metadata: dict[str, object] = {
                "profile": normalized_profile_name,
                "document_id": document.document_id,
                "document_title": document.title,
                "section": chunk.section,
            }
            metadata.update(_metadata_for_sources(document.sources))
            item_id = _build_item_id(
                profile_name=normalized_profile_name,
                document_id=document.document_id,
                section=chunk.section,
                chunk_index=chunk_index,
            )
            records.append(
                MemoryWriteRecord(
                    item_id=item_id,
                    content=chunk.content,
                    metadata=metadata,
                )
            )
            nodes, edges = extract_graph_records_from_text(chunk.content)
            _merge_graph_records(
                nodes=nodes,
                edges=edges,
                metadata=metadata,
                graph_nodes_by_id=graph_nodes_by_id,
                graph_edges_by_id=graph_edges_by_id,
            )

    resolved_sources = (
        tuple(_normalize_source(source) for source in sources) if sources else _collect_sources(normalized_documents)
    )
    return KnowledgeProfile(
        name=normalized_profile_name,
        description=normalized_description,
        records=tuple(records),
        graph_nodes=tuple(sorted(graph_nodes_by_id.values(), key=lambda node: node.node_id)),
        graph_edges=tuple(
            sorted(
                graph_edges_by_id.values(),
                key=lambda edge: (edge.source_id, edge.relationship, edge.target_id, edge.edge_id or ""),
            )
        ),
        sources=resolved_sources,
    )


def _normalize_document(document: KnowledgeDocument) -> KnowledgeDocument:
    """Validate and normalize one ingestion document."""
    document_id = document.document_id.strip()
    title = document.title.strip()
    content = document.content.strip()
    if not document_id:
        raise ValueError("KnowledgeDocument.document_id must be non-empty.")
    if not title:
        raise ValueError("KnowledgeDocument.title must be non-empty.")
    if not content:
        raise ValueError(f"KnowledgeDocument '{document_id}' must have non-empty content.")
    normalized_sources = tuple(_normalize_source(source) for source in document.sources)
    return KnowledgeDocument(
        document_id=document_id,
        title=title,
        content=content,
        sources=normalized_sources,
    )


def _chunk_markdown_document(document: KnowledgeDocument) -> tuple[_MarkdownChunk, ...]:
    """Split a Markdown document into deterministic chunks."""
    normalized_text = document.content.replace("\r\n", "\n").strip()
    matches = list(_SECTION_HEADING_PATTERN.finditer(normalized_text))
    if matches:
        chunks: list[_MarkdownChunk] = []
        preface = normalized_text[: matches[0].start()].strip()
        if preface:
            preface_content = _strip_leading_title(preface)
            if preface_content:
                chunks.append(_MarkdownChunk(section=document.title, content=preface_content))
        for index, match in enumerate(matches):
            section = match.group(1).strip()
            body_start = match.end()
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
            body = normalized_text[body_start:body_end].strip()
            rendered = _render_chunk_content(section=section, body=body)
            if rendered:
                chunks.append(_MarkdownChunk(section=section, content=rendered))
        if chunks:
            return tuple(chunks)

    paragraph_source = _strip_leading_title(normalized_text)
    paragraphs = [
        paragraph.strip() for paragraph in _PARAGRAPH_BREAK_PATTERN.split(paragraph_source) if paragraph.strip()
    ]
    if not paragraphs:
        paragraphs = [normalized_text]
    return tuple(
        _MarkdownChunk(
            section="",
            content=paragraph,
        )
        for paragraph in paragraphs
    )


def _strip_leading_title(text: str) -> str:
    """Remove one leading top-level Markdown title when present."""
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("# "):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def _render_chunk_content(*, section: str, body: str) -> str:
    """Render one section chunk with its heading included in the content."""
    normalized_body = body.strip()
    normalized_section = section.strip()
    if normalized_section and normalized_body:
        return f"{normalized_section}\n{normalized_body}"
    if normalized_body:
        return normalized_body
    return normalized_section


def _build_item_id(*, profile_name: str, document_id: str, section: str, chunk_index: int) -> str:
    """Return a deterministic item id for one ingested chunk."""
    section_label = section.strip() or f"chunk-{chunk_index:03d}"
    return f"{_slugify(profile_name)}:{_slugify(document_id)}:{_slugify(section_label)}:{chunk_index:03d}"


def _merge_graph_records(
    *,
    nodes: Sequence[GraphNodeRecord],
    edges: Sequence[GraphEdgeRecord],
    metadata: dict[str, object],
    graph_nodes_by_id: dict[str, GraphNodeRecord],
    graph_edges_by_id: dict[str, GraphEdgeRecord],
) -> None:
    """Merge extracted graph data while preserving deterministic provenance."""
    for node in nodes:
        graph_nodes_by_id.setdefault(
            node.node_id,
            GraphNodeRecord(
                node_id=node.node_id,
                name=node.name,
                node_type=node.node_type,
                description=node.description,
                metadata=dict(metadata),
            ),
        )
    for edge in edges:
        edge_id = edge.edge_id or f"{edge.source_id}:{edge.relationship}:{edge.target_id}"
        edge_metadata = dict(metadata)
        edge_metadata.update(edge.metadata)
        graph_edges_by_id.setdefault(
            edge_id,
            GraphEdgeRecord(
                source_id=edge.source_id,
                target_id=edge.target_id,
                relationship=edge.relationship,
                edge_id=edge_id,
                metadata=edge_metadata,
            ),
        )


def _normalize_source(source: KnowledgeSource) -> KnowledgeSource:
    """Validate and normalize one provenance source."""
    label = source.label.strip()
    uri = source.uri.strip()
    kind = source.kind.strip() or "unspecified"
    if not label and not uri:
        raise ValueError("KnowledgeSource must define at least one of label or uri.")
    return KnowledgeSource(label=label, uri=uri, kind=kind)


def _metadata_for_sources(sources: Sequence[KnowledgeSource]) -> dict[str, object]:
    """Return per-record metadata derived from structured provenance sources."""
    metadata: dict[str, object] = {}
    source_labels = [source.label for source in sources if source.label]
    source_uris = [source.uri for source in sources if source.uri]
    source_kinds = [source.kind for source in sources if source.kind and source.kind != "unspecified"]
    if source_labels:
        metadata["source_labels"] = list(source_labels)
    if source_uris:
        metadata["source_uris"] = list(source_uris)
    if source_kinds:
        metadata["source_kinds"] = list(source_kinds)
    if len(source_labels) == 1:
        metadata["source_label"] = source_labels[0]
    if len(source_uris) == 1:
        metadata["source_uri"] = source_uris[0]
    if len(source_kinds) == 1:
        metadata["source_kind"] = source_kinds[0]
    return metadata


def _collect_sources(documents: Sequence[KnowledgeDocument]) -> tuple[KnowledgeSource, ...]:
    """Collect stable provenance sources from ingested documents."""
    collected: list[KnowledgeSource] = []
    seen: set[tuple[str, str, str]] = set()
    for document in documents:
        for source in document.sources:
            key = (source.label, source.uri, source.kind)
            if key in seen:
                continue
            seen.add(key)
            collected.append(source)
    return tuple(collected)


def _slugify(value: str) -> str:
    """Return a deterministic slug for ids and chunk names."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "chunk"


__all__ = ["ingest_knowledge_documents"]
