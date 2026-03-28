"""Internal loaders and materializers for document-backed knowledge resources."""

from __future__ import annotations

import shutil
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath

from design_research_agents._memory._knowledge_ingestion import ingest_knowledge_documents
from design_research_agents._memory._knowledge_profile_types import (
    KnowledgeDocument,
    KnowledgeProfile,
    KnowledgeSource,
)

_RESOURCE_PACKAGE = "design_research_agents._memory._knowledge_resources"


@dataclass(slots=True, frozen=True, kw_only=True)
class _KnowledgeSource:
    """One provenance entry declared in a knowledge manifest."""

    label: str
    uri: str
    kind: str


@dataclass(slots=True, frozen=True, kw_only=True)
class _KnowledgeManifestDocument:
    """One document entry declared in a knowledge manifest."""

    document_id: str
    title: str
    path: str
    source_uri: str
    source_kind: str


@dataclass(slots=True, frozen=True, kw_only=True)
class _KnowledgeManifest:
    """Parsed knowledge manifest definition."""

    name: str
    description: str
    documents: tuple[_KnowledgeManifestDocument, ...]
    sources: tuple[_KnowledgeSource, ...]


def list_packaged_knowledge_profile_names() -> tuple[str, ...]:
    """Return packaged built-in knowledge profile names."""
    resource_root = files(_RESOURCE_PACKAGE)
    names = sorted(
        child.name for child in resource_root.iterdir() if child.is_dir() and child.joinpath("profile.toml").is_file()
    )
    return tuple(names)


@cache
def load_packaged_knowledge_profile(profile_name: str) -> KnowledgeProfile:
    """Load one packaged built-in knowledge profile from bundled resources."""
    normalized_name = _normalize_profile_name(profile_name)
    resource_root = files(_RESOURCE_PACKAGE)
    available = list_packaged_knowledge_profile_names()
    profile_dir = resource_root.joinpath(normalized_name)
    if not profile_dir.is_dir():
        available_text = ", ".join(available)
        raise ValueError(f"Unknown knowledge profile '{profile_name}'. Available profiles: {available_text}.")

    manifest_resource = profile_dir.joinpath("profile.toml")
    if not manifest_resource.is_file():
        raise ValueError(f"Packaged knowledge profile '{normalized_name}' is missing profile.toml.")

    manifest = _parse_manifest(
        manifest_text=manifest_resource.read_text(encoding="utf-8"),
        source_label=f"packaged knowledge profile '{normalized_name}'",
    )
    documents = tuple(
        KnowledgeDocument(
            document_id=document.document_id,
            title=document.title,
            content=_read_packaged_document(profile_dir=profile_dir, relative_path=document.path),
            sources=_build_document_sources(document, manifest.sources),
        )
        for document in manifest.documents
    )
    return ingest_knowledge_documents(
        manifest.name,
        description=manifest.description,
        documents=documents,
        sources=_format_sources(manifest.sources),
    )


def load_source_knowledge_profile(profile_name: str, *, source_root: Path) -> KnowledgeProfile:
    """Load one source knowledge profile from the repo-local ``knowledge/`` tree."""
    manifest, profile_dir = load_source_manifest(profile_name, source_root=source_root)
    documents = tuple(
        KnowledgeDocument(
            document_id=document.document_id,
            title=document.title,
            content=_resolve_source_document_text(profile_dir=profile_dir, relative_path=document.path),
            sources=_build_document_sources(document, manifest.sources),
        )
        for document in manifest.documents
    )
    return ingest_knowledge_documents(
        manifest.name,
        description=manifest.description,
        documents=documents,
        sources=_format_sources(manifest.sources),
    )


def load_source_manifest(profile_name: str, *, source_root: Path) -> tuple[_KnowledgeManifest, Path]:
    """Load and validate one repo-local source manifest."""
    normalized_name = _normalize_profile_name(profile_name)
    profile_dir = source_root / normalized_name
    manifest_path = profile_dir / "profile.toml"
    if not manifest_path.is_file():
        available = ", ".join(sorted(path.name for path in source_root.iterdir() if path.is_dir()))
        raise ValueError(f"Unknown knowledge profile '{profile_name}'. Available profiles: {available}.")
    manifest = _parse_manifest(
        manifest_text=manifest_path.read_text(encoding="utf-8"),
        source_label=str(manifest_path),
    )
    if manifest.name != normalized_name:
        raise ValueError(f"{manifest_path}: manifest name '{manifest.name}' must match directory '{normalized_name}'.")
    for document in manifest.documents:
        _resolve_source_document_path(profile_dir=profile_dir, relative_path=document.path)
    return manifest, profile_dir


def materialize_source_knowledge_profile(
    profile_name: str,
    *,
    source_root: Path,
    packaged_root: Path,
) -> Path:
    """Copy one repo-local source profile into packaged runtime resources."""
    manifest, profile_dir = load_source_manifest(profile_name, source_root=source_root)
    destination_dir = packaged_root / manifest.name
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = profile_dir / "profile.toml"
    destination_manifest_path = destination_dir / "profile.toml"
    destination_manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").rstrip() + "\n",
        encoding="utf-8",
    )

    for document in manifest.documents:
        source_document_path = _resolve_source_document_path(profile_dir=profile_dir, relative_path=document.path)
        destination_document_path = destination_dir / Path(document.path)
        destination_document_path.parent.mkdir(parents=True, exist_ok=True)
        destination_document_path.write_text(
            source_document_path.read_text(encoding="utf-8").rstrip() + "\n",
            encoding="utf-8",
        )
    return destination_dir


def _parse_manifest(*, manifest_text: str, source_label: str) -> _KnowledgeManifest:
    """Parse one TOML knowledge manifest."""
    data = tomllib.loads(manifest_text)
    if not isinstance(data, dict):
        raise ValueError(f"{source_label}: manifest must deserialize into a table.")

    name = _get_required_string(data, key="name", source_label=source_label)
    description = _get_required_string(data, key="description", source_label=source_label)
    documents_raw = data.get("documents")
    if not isinstance(documents_raw, list) or not documents_raw:
        raise ValueError(f"{source_label}: manifest must define at least one [[documents]] entry.")

    sources_raw = data.get("sources", [])
    if not isinstance(sources_raw, list):
        raise ValueError(f"{source_label}: sources must be declared with [[sources]] tables.")

    documents = tuple(_parse_document(entry, source_label=source_label) for entry in documents_raw)
    sources = tuple(_parse_source(entry, source_label=source_label) for entry in sources_raw)
    _validate_sources(sources, source_label=source_label)
    _validate_document_sources(documents=documents, sources=sources, source_label=source_label)
    return _KnowledgeManifest(
        name=name,
        description=description,
        documents=documents,
        sources=sources,
    )


def _parse_document(entry: object, *, source_label: str) -> _KnowledgeManifestDocument:
    """Parse one manifest document entry."""
    if not isinstance(entry, dict):
        raise ValueError(f"{source_label}: each [[documents]] entry must be a table.")
    document_id = _get_required_string(entry, key="id", source_label=source_label)
    title = _get_required_string(entry, key="title", source_label=source_label)
    path = _get_required_string(entry, key="path", source_label=source_label)
    source_uri = _get_required_string(entry, key="source_uri", source_label=source_label)
    source_kind = _get_required_string(entry, key="source_kind", source_label=source_label)
    _validate_relative_path(path, source_label=source_label)
    if not path.endswith(".md"):
        raise ValueError(f"{source_label}: document path '{path}' must point to a Markdown file.")
    return _KnowledgeManifestDocument(
        document_id=document_id,
        title=title,
        path=path,
        source_uri=source_uri,
        source_kind=source_kind,
    )


def _parse_source(entry: object, *, source_label: str) -> _KnowledgeSource:
    """Parse one manifest provenance source entry."""
    if not isinstance(entry, dict):
        raise ValueError(f"{source_label}: each [[sources]] entry must be a table.")
    label = _get_required_string(entry, key="label", source_label=source_label)
    uri = _get_required_string(entry, key="uri", source_label=source_label)
    kind = _get_optional_string(entry, key="kind", source_label=source_label) or "unspecified"
    return _KnowledgeSource(label=label, uri=uri, kind=kind)


def _get_required_string(mapping: dict[str, object], *, key: str, source_label: str) -> str:
    """Read one required string value from a manifest mapping."""
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source_label}: '{key}' must be a non-empty string.")
    return value.strip()


def _get_optional_string(mapping: dict[str, object], *, key: str, source_label: str) -> str:
    """Read one optional trimmed string value from a manifest mapping."""
    value = mapping.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{source_label}: '{key}' must be a string when provided.")
    return value.strip()


def _normalize_profile_name(profile_name: str) -> str:
    """Normalize profile name lookup keys."""
    normalized_name = profile_name.strip().lower()
    if not normalized_name:
        raise ValueError("profile_name must be non-empty.")
    return normalized_name


def _validate_relative_path(path_text: str, *, source_label: str) -> PurePosixPath:
    """Validate one relative manifest path."""
    relative_path = PurePosixPath(path_text)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{source_label}: path '{path_text}' must stay within the profile directory.")
    if not relative_path.parts:
        raise ValueError(f"{source_label}: path '{path_text}' must be non-empty.")
    return relative_path


def _resolve_source_document_path(*, profile_dir: Path, relative_path: str) -> Path:
    """Resolve one repo-local source document path and ensure it is contained."""
    validated_relative_path = _validate_relative_path(relative_path, source_label=str(profile_dir / "profile.toml"))
    resolved_path = (profile_dir / Path(*validated_relative_path.parts)).resolve()
    profile_dir_resolved = profile_dir.resolve()
    try:
        resolved_path.relative_to(profile_dir_resolved)
    except ValueError as exc:
        raise ValueError(
            f"{profile_dir / 'profile.toml'}: path '{relative_path}' must stay within the profile directory."
        ) from exc
    if not resolved_path.is_file():
        raise ValueError(f"{profile_dir / 'profile.toml'}: document path '{relative_path}' does not exist.")
    return resolved_path


def _resolve_source_document_text(*, profile_dir: Path, relative_path: str) -> str:
    """Read one repo-local source document after pointer validation."""
    return (
        _resolve_source_document_path(profile_dir=profile_dir, relative_path=relative_path)
        .read_text(encoding="utf-8")
        .strip()
    )


def _read_packaged_document(*, profile_dir: Traversable, relative_path: str) -> str:
    """Read one packaged Markdown document."""
    validated_relative_path = _validate_relative_path(
        relative_path,
        source_label=f"{profile_dir.name}/profile.toml",
    )
    resource = profile_dir.joinpath(*validated_relative_path.parts)
    if not resource.is_file():
        raise ValueError(f"Packaged knowledge profile '{profile_dir.name}' is missing document '{relative_path}'.")
    return str(resource.read_text(encoding="utf-8")).strip()


def _format_sources(sources: Sequence[_KnowledgeSource]) -> tuple[KnowledgeSource, ...]:
    """Convert manifest provenance entries into structured public payloads."""
    return tuple(
        KnowledgeSource(
            label=source.label,
            uri=source.uri,
            kind=source.kind,
        )
        for source in sources
    )


def _build_document_sources(
    document: _KnowledgeManifestDocument,
    manifest_sources: Sequence[_KnowledgeSource],
) -> tuple[KnowledgeSource, ...]:
    """Build structured document provenance from manifest fields."""
    matching_source = next((source for source in manifest_sources if source.uri == document.source_uri), None)
    source_label = matching_source.label if matching_source is not None else ""
    source_kind = document.source_kind or (matching_source.kind if matching_source is not None else "unspecified")
    return (
        KnowledgeSource(
            label=source_label,
            uri=document.source_uri,
            kind=source_kind,
        ),
    )


def _validate_sources(sources: Sequence[_KnowledgeSource], *, source_label: str) -> None:
    """Validate source declarations needed for deterministic provenance mapping."""
    seen_uris: set[str] = set()
    for source in sources:
        if source.uri in seen_uris:
            raise ValueError(f"{source_label}: duplicate source uri '{source.uri}' is not allowed.")
        seen_uris.add(source.uri)


def _validate_document_sources(
    *,
    documents: Sequence[_KnowledgeManifestDocument],
    sources: Sequence[_KnowledgeSource],
    source_label: str,
) -> None:
    """Ensure document-level provenance points at declared manifest sources when present."""
    if not sources:
        return

    source_uris = {source.uri for source in sources}
    for document in documents:
        if document.source_uri not in source_uris:
            raise ValueError(
                f"{source_label}: document '{document.document_id}' references "
                f"unknown source_uri '{document.source_uri}'."
            )


__all__ = [
    "list_packaged_knowledge_profile_names",
    "load_packaged_knowledge_profile",
    "load_source_knowledge_profile",
    "load_source_manifest",
    "materialize_source_knowledge_profile",
]
