from __future__ import annotations

from pathlib import Path

import pytest

from design_research_agents._memory import _knowledge_resource_loader as loader


def _manifest(
    *,
    name: str = "demo",
    document_path: str = "notes/design.md",
    source_uri: str = "https://example.invalid/source",
    include_sources: bool = True,
) -> str:
    source_block = (
        "\n".join(
            [
                "[[sources]]",
                'label = "Reference"',
                f'uri = "{source_uri}"',
                'kind = "background_reference"',
                "",
            ]
        )
        if include_sources
        else ""
    )
    return "\n".join(
        [
            f'name = "{name}"',
            'description = "Demo profile"',
            "",
            source_block,
            "[[documents]]",
            'id = "design-notes"',
            'title = "Design Notes"',
            f'path = "{document_path}"',
            f'source_uri = "{source_uri}"',
            'source_kind = "curated_note"',
            "",
        ]
    )


def _write_source_profile(source_root: Path, *, manifest: str | None = None) -> Path:
    profile_dir = source_root / "demo"
    document_path = profile_dir / "notes" / "design.md"
    document_path.parent.mkdir(parents=True)
    document_path.write_text("# Design Notes\n\nA depends on B.\n", encoding="utf-8")
    (profile_dir / "profile.toml").write_text(manifest or _manifest(), encoding="utf-8")
    return profile_dir


def test_source_profile_load_and_materialization_preserve_content_and_provenance(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    profile_dir = _write_source_profile(source_root)
    packaged_root = tmp_path / "packaged"
    stale_destination = packaged_root / "demo"
    stale_destination.mkdir(parents=True)
    (stale_destination / "stale.txt").write_text("stale", encoding="utf-8")

    profile = loader.load_source_knowledge_profile(" DEMO ", source_root=source_root)
    destination = loader.materialize_source_knowledge_profile(
        "demo",
        source_root=source_root,
        packaged_root=packaged_root,
    )

    assert profile.name == "demo"
    assert profile.records[0].metadata["source_label"] == "Reference"
    assert profile.sources[0].kind == "background_reference"
    assert destination == packaged_root / "demo"
    assert not (destination / "stale.txt").exists()
    assert (destination / "profile.toml").read_text(encoding="utf-8").endswith("\n")
    assert (destination / "notes" / "design.md").read_text(encoding="utf-8").endswith("\n")
    assert profile_dir.is_dir()


def test_source_manifest_reports_unknown_name_mismatch_and_missing_document(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "available").mkdir(parents=True)
    with pytest.raises(ValueError, match="Available profiles: available"):
        loader.load_source_manifest("missing", source_root=source_root)

    profile_dir = _write_source_profile(source_root, manifest=_manifest(name="different"))
    with pytest.raises(ValueError, match="must match directory"):
        loader.load_source_manifest("demo", source_root=source_root)

    (profile_dir / "profile.toml").write_text(_manifest(document_path="missing.md"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        loader.load_source_manifest("demo", source_root=source_root)


@pytest.mark.parametrize(
    ("manifest_text", "message"),
    [
        ('name = "demo"\ndescription = "Demo"\n', "at least one"),
        (
            'name = "demo"\ndescription = "Demo"\nsources = "invalid"\n[[documents]]\n'
            'id = "a"\ntitle = "A"\npath = "a.md"\nsource_uri = "u"\nsource_kind = "k"\n',
            "sources must be declared",
        ),
        (
            'name = "demo"\ndescription = "Demo"\ndocuments = ["invalid"]\n',
            r"each \[\[documents\]\] entry must be a table",
        ),
        (
            'name = "demo"\ndescription = "Demo"\n[[documents]]\nid = ""\n',
            "'id' must be a non-empty string",
        ),
        (_manifest(document_path="notes.txt"), "must point to a Markdown file"),
        (
            _manifest() + '\n[[sources]]\nlabel = "Duplicate"\nuri = "https://example.invalid/source"\n',
            "duplicate source uri",
        ),
        (
            _manifest(source_uri="https://example.invalid/unknown").replace(
                'uri = "https://example.invalid/unknown"',
                'uri = "https://example.invalid/declared"',
                1,
            ),
            "unknown source_uri",
        ),
    ],
)
def test_manifest_validation_reports_actionable_errors(manifest_text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        loader._parse_manifest(manifest_text=manifest_text, source_label="profile.toml")


def test_manifest_source_and_scalar_validation_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader.tomllib, "loads", lambda _text: [])
    with pytest.raises(ValueError, match="must deserialize into a table"):
        loader._parse_manifest(manifest_text="ignored", source_label="profile.toml")

    with pytest.raises(ValueError, match=r"each \[\[sources\]\] entry must be a table"):
        loader._parse_source("invalid", source_label="profile.toml")
    with pytest.raises(ValueError, match="must be a string when provided"):
        loader._get_optional_string({"kind": 3}, key="kind", source_label="profile.toml")
    assert loader._get_optional_string({}, key="kind", source_label="profile.toml") == ""
    with pytest.raises(ValueError, match="profile_name must be non-empty"):
        loader._normalize_profile_name(" ")


@pytest.mark.parametrize("path_text", ["/absolute.md", "../escape.md", ""])
def test_relative_path_validation_rejects_unsafe_or_empty_paths(path_text: str) -> None:
    with pytest.raises(ValueError, match=r"must (stay within|be non-empty)"):
        loader._validate_relative_path(path_text, source_label="profile.toml")


def test_source_document_symlink_cannot_escape_profile(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    symlink = profile_dir / "linked.md"
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="must stay within"):
        loader._resolve_source_document_path(profile_dir=profile_dir, relative_path="linked.md")


def test_packaged_profile_errors_and_fallback_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_root = tmp_path / "resources"
    resource_root.mkdir()
    monkeypatch.setattr(loader, "files", lambda _package: resource_root)
    loader.load_packaged_knowledge_profile.cache_clear()

    with pytest.raises(ValueError, match="Unknown knowledge profile"):
        loader.load_packaged_knowledge_profile("missing")

    profile_dir = resource_root / "demo"
    profile_dir.mkdir()
    with pytest.raises(ValueError, match=r"missing profile\.toml"):
        loader.load_packaged_knowledge_profile("demo")

    (profile_dir / "profile.toml").write_text(_manifest(include_sources=False), encoding="utf-8")
    loader.load_packaged_knowledge_profile.cache_clear()
    with pytest.raises(ValueError, match="missing document"):
        loader.load_packaged_knowledge_profile("demo")

    document_path = profile_dir / "notes" / "design.md"
    document_path.parent.mkdir(parents=True)
    document_path.write_text("# Notes\n\nA uses B.\n", encoding="utf-8")
    loader.load_packaged_knowledge_profile.cache_clear()
    profile = loader.load_packaged_knowledge_profile("demo")
    assert profile.sources[0].label == ""
    manifest, _ = loader.load_source_manifest(
        "demo",
        source_root=resource_root,
    )
    assert loader._build_document_sources(manifest.documents[0], ())[0].label == ""
    loader.load_packaged_knowledge_profile.cache_clear()
