from __future__ import annotations

import builtins
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from design_research_agents._memory import _knowledge_resource_loader as loader
from design_research_agents._model_selection import _hardware as hw
from design_research_agents._model_selection._catalog import (
    ModelCatalog,
    ModelFlight,
    ModelFlightRegistry,
    _artifact_sort_key,
    _call_huggingface_model_info,
    _card_data_mapping,
    _huggingface_context_window,
    _huggingface_info_tags,
    _huggingface_license,
    _infer_family_from_repo_id,
    _infer_model_format_from_artifact,
    _infer_quantization_from_labels,
    _infer_size_b_from_labels,
    _load_huggingface_api,
    _model_spec_from_huggingface_info,
    _preferred_huggingface_artifact,
)
from design_research_agents._model_selection._types import ModelSpec


def _write_source_profile(root, name: str = "demo") -> None:
    profile_dir = root / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "docs").mkdir()
    (profile_dir / "docs" / "one.md").write_text("# One\n\nUseful knowledge.\n", encoding="utf-8")
    (profile_dir / "profile.toml").write_text(
        """
name = "demo"
description = "Demo profile."

[[sources]]
label = "Manual"
uri = "manual://one"
kind = "reference"

[[documents]]
id = "doc-one"
title = "One"
path = "docs/one.md"
source_uri = "manual://one"
source_kind = "reference"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_source_knowledge_profiles_load_materialize_and_validate_edges(tmp_path) -> None:
    source_root = tmp_path / "knowledge"
    packaged_root = tmp_path / "packaged"
    _write_source_profile(source_root)
    (packaged_root / "demo").mkdir(parents=True)
    (packaged_root / "demo" / "stale.txt").write_text("old", encoding="utf-8")

    manifest, profile_dir = loader.load_source_manifest(" DEMO ", source_root=source_root)
    profile = loader.load_source_knowledge_profile("demo", source_root=source_root)
    materialized = loader.materialize_source_knowledge_profile(
        "demo",
        source_root=source_root,
        packaged_root=packaged_root,
    )

    assert manifest.name == "demo"
    assert profile_dir == source_root / "demo"
    assert profile.name == "demo"
    assert profile.sources[0].label == "Manual"
    assert materialized == packaged_root / "demo"
    assert not (materialized / "stale.txt").exists()
    assert (materialized / "docs" / "one.md").read_text(encoding="utf-8").endswith("\n")

    with pytest.raises(ValueError, match="profile_name must be non-empty"):
        loader.load_source_manifest(" ", source_root=source_root)
    with pytest.raises(ValueError, match="Unknown knowledge profile"):
        loader.load_source_manifest("missing", source_root=source_root)

    bad_root = tmp_path / "bad"
    _write_source_profile(bad_root)
    (bad_root / "demo" / "profile.toml").write_text(
        'name = "wrong"\ndescription = "Bad."\n[[documents]]\n'
        'id = "doc"\ntitle = "Doc"\npath = "docs/one.md"\n'
        'source_uri = "manual://one"\nsource_kind = "reference"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must match directory"):
        loader.load_source_manifest("demo", source_root=bad_root)


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ('name = "demo"\ndescription = "Demo."\n', "at least one"),
        ('name = "demo"\ndescription = "Demo."\nsources = "bad"\n[[documents]]\n', "sources must be declared"),
        (
            'name = "demo"\ndescription = "Demo."\n[[documents]]\n'
            'id = "doc"\ntitle = "Doc"\npath = "../escape.md"\n'
            'source_uri = "u"\nsource_kind = "reference"\n',
            "must stay within",
        ),
        (
            'name = "demo"\ndescription = "Demo."\n[[documents]]\n'
            'id = "doc"\ntitle = "Doc"\npath = "doc.txt"\nsource_uri = "u"\nsource_kind = "reference"\n',
            "Markdown",
        ),
        (
            'name = "demo"\ndescription = "Demo."\n[[sources]]\nlabel = "A"\nuri = "u"\nkind = 3\n'
            '[[documents]]\nid = "doc"\ntitle = "Doc"\npath = "doc.md"\nsource_uri = "u"\nsource_kind = "reference"\n',
            "kind",
        ),
        (
            'name = "demo"\ndescription = "Demo."\n[[sources]]\nlabel = "A"\nuri = "u"\n'
            '[[sources]]\nlabel = "B"\nuri = "u"\n[[documents]]\n'
            'id = "doc"\ntitle = "Doc"\npath = "doc.md"\nsource_uri = "u"\nsource_kind = "reference"\n',
            "duplicate source uri",
        ),
        (
            'name = "demo"\ndescription = "Demo."\n[[sources]]\nlabel = "A"\nuri = "u"\n'
            '[[documents]]\nid = "doc"\ntitle = "Doc"\npath = "doc.md"\n'
            'source_uri = "unknown"\nsource_kind = "reference"\n',
            "unknown source_uri",
        ),
    ],
)
def test_manifest_parser_rejects_invalid_shapes(manifest: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        loader._parse_manifest(manifest_text=manifest, source_label="profile.toml")


def test_manifest_private_helpers_cover_missing_document_and_packaged_paths(tmp_path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()

    with pytest.raises(ValueError, match="does not exist"):
        loader._resolve_source_document_path(profile_dir=profile_dir, relative_path="missing.md")

    class _Resource:
        name = "demo"

        def __init__(self, files: dict[tuple[str, ...], str], path: tuple[str, ...] = ()) -> None:
            self.files = files
            self.path = path

        def joinpath(self, *parts: str) -> _Resource:
            return _Resource(self.files, (*self.path, *parts))

        def is_file(self) -> bool:
            return self.path in self.files

        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return self.files[self.path]

    resource = _Resource({("docs", "one.md"): " packaged \n"})
    assert loader._read_packaged_document(profile_dir=resource, relative_path="docs/one.md") == "packaged"
    with pytest.raises(ValueError, match="missing document"):
        loader._read_packaged_document(profile_dir=resource, relative_path="docs/missing.md")

    document = loader._KnowledgeManifestDocument(
        document_id="doc",
        title="Doc",
        path="doc.md",
        source_uri="direct://source",
        source_kind="",
    )
    assert loader._build_document_sources(document, ())[0].kind == "unspecified"


def _model(model_id: str) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        provider="test",
        family="demo",
        size_b=None,
        format="api",
        quantization=None,
        memory_hint=None,
        latency_hint=None,
        cost_hint=None,
        quality_tier=None,
        speed_tier=None,
    )


def test_model_catalog_registry_and_huggingface_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model("m")
    flight = ModelFlight(flight_id=" f ", description=" d ", models=(model,), tags=(" keep ", ""))
    assert flight.flight_id == "f"
    assert flight.description == "d"
    assert flight.tags == ("keep",)
    assert flight.with_models((model,)).model_ids() == ("m",)

    registry = ModelFlightRegistry(flights=(flight,))
    assert registry.find("missing") is None
    with pytest.raises(KeyError, match="Unknown model flight"):
        registry.require("missing")
    with pytest.raises(ValueError, match="flights must contain"):
        ModelFlightRegistry(flights=())

    catalog = ModelCatalog(models=(model,))
    assert catalog.signature()
    with pytest.raises(KeyError, match="Unknown model"):
        catalog.require("missing")
    assert catalog.filter(provider="other").models == ()
    assert catalog.filter(min_size_b=1.0).models == ()
    assert catalog.filter(max_size_b=1.0).models == ()
    with pytest.raises(ValueError, match="repo_ids"):
        ModelCatalog.from_huggingface((" ",), api=object())

    class _BadApi:
        model_info = "not callable"

    with pytest.raises(TypeError, match="model_info"):
        _call_huggingface_model_info(_BadApi(), repo_id="repo", revision=None, token=None, timeout=None)

    real_import = builtins.__import__

    def _missing_hf(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_hf)
    with pytest.raises(ImportError, match="huggingface_hub"):
        _load_huggingface_api(token=None)


@dataclass(frozen=True)
class _Sibling:
    rfilename: str


def test_huggingface_metadata_helpers_cover_fallbacks() -> None:
    assert _huggingface_info_tags(SimpleNamespace(tags="text-generation")) == ()
    assert _preferred_huggingface_artifact(SimpleNamespace(siblings=object())) is None
    assert _preferred_huggingface_artifact(SimpleNamespace(siblings=())) is None
    assert _artifact_sort_key("weights.bin") == (2, "weights.bin")
    assert _artifact_sort_key("README.md") == (3, "README.md")
    assert _infer_model_format_from_artifact(None) is None
    assert _infer_model_format_from_artifact("pytorch_model.bin") == "pytorch"
    assert _infer_model_format_from_artifact("README.md") is None
    assert _infer_quantization_from_labels(("model-q6_k",)) == "q6_k"
    assert _infer_quantization_from_labels(("model",)) is None
    assert _infer_size_b_from_labels(("mixtral-8x7b",)) == 56.0
    assert _infer_size_b_from_labels(("model",)) is None
    assert _infer_family_from_repo_id("/") == ""
    assert _huggingface_license(SimpleNamespace(cardData={"license": 1})) is None
    assert _huggingface_license(SimpleNamespace()) is None
    assert _huggingface_context_window(SimpleNamespace(config={"seq_length": 2048})) == 2048
    assert _huggingface_context_window(SimpleNamespace(config={"seq_length": 0})) is None
    assert _card_data_mapping(SimpleNamespace(card_data={"license": "x"})) == {"license": "x"}

    info = SimpleNamespace(
        tags=("q8_0", 7),
        siblings=(_Sibling("model.safetensors"), _Sibling("model.gguf")),
        card_data={"license": "demo"},
        config={"n_positions": 4096},
        author="alice",
        private=False,
        gated=True,
        downloads_all_time=12,
        sha=" ",
    )
    spec = _model_spec_from_huggingface_info(
        info,
        repo_id="Org/Model-1B",
        provider="openai",
        family="family",
        revision="main",
        model_format="custom",
        capabilities=("chat", "chat"),
        tags=("tag",),
        quality_tier=4,
        speed_tier=5,
    )
    assert spec.format == "custom"
    assert spec.revision == "main"
    assert spec.license == "demo"
    assert spec.metadata["huggingface_author"] == "alice"
    assert spec.tags == ("tag", "q8_0", "7")


def test_hardware_profile_and_remaining_detection_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = hw.HardwareProfile(
        total_ram_gb=1.0,
        available_ram_gb=0.5,
        cpu_count=2,
        load_average=(1.0, 2.0, 3.0),
        gpu_present=True,
        gpu_vram_gb=4.0,
        gpu_name="GPU",
        platform_name="TestOS",
    )
    assert profile.to_dict()["gpu_name"] == "GPU"
    assert '"gpu_name": "GPU"' in str(profile)

    monkeypatch.setattr(hw.os, "sysconf", lambda _key: (_ for _ in ()).throw(ValueError("bad")))
    assert hw._detect_sysconf_total_ram_bytes() is None

    monkeypatch.setattr(hw, "_run_command", lambda _args: "bad-row\nGPU, bad\n")
    assert hw._detect_nvidia_gpu_info() == (True, 0, "GPU")

    monkeypatch.setattr(hw, "_run_command", lambda _args: "Chipset Model: Only Name\nNo digits: none")
    assert hw._detect_macos_gpu_info() == (True, None, "Only Name")

    monkeypatch.setattr(hw, "_run_command", lambda _args: "HOST,bad,GPU\nHOST,123,Other\n")
    assert hw._detect_windows_gpu_info() == (True, 123, "GPU")

    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_windows_memory_status", lambda: SimpleNamespace(ullTotalPhys=10, ullAvailPhys=4))
    assert hw._detect_windows_total_ram_bytes() == 10
    assert hw._detect_windows_available_ram_bytes() == 4
