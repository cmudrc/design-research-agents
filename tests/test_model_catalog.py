from __future__ import annotations

import builtins
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from design_research_agents._model_selection import _catalog as catalog_impl
from design_research_agents._model_selection._catalog import (
    ModelCatalog,
    ModelFlight,
    ModelFlightRegistry,
)
from design_research_agents._model_selection._default_flights import (
    _estimate_gguf_memory_hint,
    _latency_tier,
    _quality_tier,
    _speed_tier,
)
from design_research_agents._model_selection._types import (
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSpec,
)


def test_default_catalog_has_expected_shape_and_is_discoverable() -> None:
    catalog = ModelCatalog.default()
    assert len(catalog.models) >= 100
    assert catalog.find("gpt-4o") is not None
    assert catalog.find("gemma-3-12b-it-gguf-q4_k_m") is not None
    assert catalog.find("llama-3.1-8b-instruct-gguf-q8_0") is not None
    assert catalog.find("openai/gpt-oss-20b") is not None
    assert catalog.find("deepseek-ai/DeepSeek-V3.2") is not None
    assert catalog.find("Qwen/Qwen3-Coder-480B-A35B-Instruct") is not None
    assert catalog.find("Qwen/Qwen3-VL-32B-Instruct") is not None
    assert catalog.find("does-not-exist") is None
    assert catalog.require("gpt-4o").source == "curated"
    assert "structured_output" in catalog.require("gpt-4o").capabilities
    assert "reasoning" in catalog.require("openai/gpt-oss-20b").capabilities
    assert "vision" in catalog.require("Qwen/Qwen3-VL-32B-Instruct").capabilities


def test_model_spec_normalizes_metadata_labels_and_context() -> None:
    spec = ModelSpec(
        model_id=" demo ",
        provider=" llama_cpp ",
        family=" qwen ",
        size_b=1.0,
        format=" gguf ",
        quantization=" q4_k_m ",
        memory_hint=ModelMemoryHint(min_ram_gb=1.0, min_vram_gb=0.5),
        latency_hint=ModelLatencyHint(tier="fast"),
        cost_hint=ModelCostHint(tier="low", usd_per_1k_tokens=0.0),
        quality_tier=1,
        speed_tier=5,
        source=" ",
        context_window=4096,
        capabilities=(" chat ", "chat", ""),
        tags=(" local ", "local", "gguf"),
        metadata={"rank": 1},
    )

    assert spec.model_id == "demo"
    assert spec.provider == "llama_cpp"
    assert spec.family == "qwen"
    assert spec.source == "curated"
    assert spec.capabilities == ("chat",)
    assert spec.tags == ("local", "gguf")
    assert spec.metadata == {"rank": 1}

    with pytest.raises(ValueError, match="context_window"):
        ModelSpec(
            model_id="bad-context",
            provider="llama_cpp",
            family="qwen",
            size_b=1.0,
            format="gguf",
            quantization="q4_k_m",
            memory_hint=None,
            latency_hint=None,
            cost_hint=None,
            quality_tier=None,
            speed_tier=None,
            context_window=0,
        )


def test_default_flight_registry_exposes_named_model_matrices() -> None:
    flight_registry = ModelFlightRegistry.default()

    assert flight_registry.flight_ids() == (
        "qwen3-gguf",
        "gemma3-gguf",
        "llama-gguf",
        "mistral-gguf",
        "phi-gguf",
        "open-reasoning",
        "frontier-moe-open-weights",
        "agentic-coding-open-weights",
        "vision-language-open-weights",
        "openai-api",
    )

    qwen = flight_registry.require("qwen3-gguf")
    assert len(qwen.models) == 24
    assert qwen.model_ids()[0] == "qwen3-0.6b-instruct-gguf-q4_k_m"
    assert qwen.model_ids()[-1] == "qwen3-32b-instruct-gguf-q8_0"
    assert {model.quantization for model in qwen.models} == {"q4_k_m", "q5_k_m", "q6_k", "q8_0"}
    assert {model.size_b for model in qwen.models} == {0.6, 1.8, 4.0, 7.0, 14.0, 32.0}

    gemma = flight_registry.require("gemma3-gguf")
    assert len(gemma.models) == 16
    assert {model.family for model in gemma.models} == {"gemma3"}

    reasoning = flight_registry.require("open-reasoning")
    assert reasoning.model_ids()[:2] == ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
    assert {model.family for model in reasoning.models} >= {"gpt-oss", "deepseek-r1", "phi4"}

    coding = flight_registry.require("agentic-coding-open-weights")
    assert "Qwen/Qwen3-Coder-480B-A35B-Instruct" in coding.model_ids()
    assert "moonshotai/Kimi-K2-Thinking" in coding.model_ids()

    vision = flight_registry.require("vision-language-open-weights")
    assert "google/gemma-3-27b-it" in vision.model_ids()
    assert "Qwen/Qwen3-VL-32B-Instruct" in vision.model_ids()


def test_catalog_can_be_built_from_selected_flights() -> None:
    flight_registry = ModelFlightRegistry.default()
    selected_flights = (
        flight_registry.require("gemma3-gguf"),
        flight_registry.require("openai-api"),
    )

    catalog = ModelCatalog.from_flights(selected_flights)

    assert len(catalog.models) == 18
    assert catalog.find("gemma-3-27b-it-gguf-q6_k") is not None
    assert catalog.find("gpt-4o-mini") is not None
    assert catalog.find("qwen3-14b-instruct-gguf-q4_k_m") is None


def test_catalog_query_helpers_filter_models() -> None:
    catalog = ModelCatalog.default()

    assert len(catalog.by_family("gemma3").filter(model_format="gguf").models) == 16
    assert len(catalog.by_provider("openai").models) == 2
    assert catalog.local().find("gpt-4o") is None
    assert catalog.remote().model_ids() == ("gpt-4o-mini", "gpt-4o")
    assert catalog.filter(family="qwen3", quantization="q4_k_m").model_ids() == (
        "qwen3-0.6b-instruct-gguf-q4_k_m",
        "qwen3-1.8b-instruct-gguf-q4_k_m",
        "qwen3-4b-instruct-gguf-q4_k_m",
        "qwen3-7b-instruct-gguf-q4_k_m",
        "qwen3-14b-instruct-gguf-q4_k_m",
        "qwen3-32b-instruct-gguf-q4_k_m",
    )
    assert len(catalog.with_tag("gemma").models) == 20
    assert len(catalog.with_capability("structured_output").models) == 2
    assert len(catalog.with_capability("reasoning").models) >= 10
    assert len(catalog.with_capability("vision").models) >= 8
    assert len(catalog.filter(min_size_b=7.0, max_size_b=8.0).models) == 13


def test_catalog_merge_handles_duplicates_explicitly() -> None:
    openai = ModelCatalog.from_flights((ModelFlightRegistry.default().require("openai-api"),))
    gemma = ModelCatalog.from_flights((ModelFlightRegistry.default().require("gemma3-gguf"),))

    merged = openai.merge(gemma)
    assert len(merged.models) == 18

    with pytest.raises(ValueError, match="Duplicate model id"):
        openai.merge(openai)

    replaced = openai.merge(openai, replace=True)
    assert replaced.model_ids() == openai.model_ids()


@dataclass(slots=True, frozen=True)
class _RepoSibling:
    rfilename: str


@dataclass(slots=True, frozen=True)
class _ModelInfo:
    id: str
    sha: str
    tags: tuple[str, ...]
    siblings: tuple[_RepoSibling, ...]
    cardData: dict[str, object]
    config: dict[str, object]
    downloads: int
    likes: int


class _FakeHuggingFaceApi:
    def model_info(self, **kwargs: object) -> _ModelInfo:
        assert kwargs["repo_id"] == "google/gemma-3-4b-it"
        assert kwargs["revision"] == "main"
        assert kwargs["files_metadata"] is True
        return _ModelInfo(
            id="google/gemma-3-4b-it",
            sha="abc123",
            tags=("text-generation", "gguf", "q4_k_m"),
            siblings=(
                _RepoSibling("README.md"),
                _RepoSibling("gemma-3-4b-it-q4_k_m.gguf"),
                _RepoSibling("model.safetensors"),
            ),
            cardData={"license": "gemma"},
            config={"max_position_embeddings": 8192},
            downloads=42,
            likes=7,
        )


def test_catalog_can_be_built_from_huggingface_metadata_without_network() -> None:
    catalog = ModelCatalog.from_huggingface(
        ("google/gemma-3-4b-it",),
        provider="llama_cpp",
        revision="main",
        api=_FakeHuggingFaceApi(),
        capabilities=("chat", "local"),
        tags=("candidate",),
    )

    model = catalog.require("google/gemma-3-4b-it")
    assert model.source == "huggingface"
    assert model.repo_id == "google/gemma-3-4b-it"
    assert model.revision == "abc123"
    assert model.family == "gemma"
    assert model.size_b == 4.0
    assert model.format == "gguf"
    assert model.quantization == "q4_k_m"
    assert model.artifact == "gemma-3-4b-it-q4_k_m.gguf"
    assert model.license == "gemma"
    assert model.context_window == 8192
    assert model.capabilities == ("chat", "local")
    assert model.tags == ("candidate", "text-generation", "gguf", "q4_k_m")
    assert model.source_url == "https://huggingface.co/google/gemma-3-4b-it"
    assert model.metadata["huggingface_downloads"] == 42


def test_model_flight_validation_rejects_empty_and_duplicate_sets() -> None:
    model = ModelFlightRegistry.default().require("openai-api").models[0]

    with pytest.raises(ValueError, match="flight_id"):
        ModelFlight(flight_id=" ", description="empty id", models=(model,))
    with pytest.raises(ValueError, match="models must contain"):
        ModelFlight(flight_id="empty", description="empty model set", models=())
    with pytest.raises(ValueError, match="duplicate model ids"):
        ModelFlight(flight_id="dupe", description="duplicate models", models=(model, model))


def test_model_flight_registry_validation_rejects_duplicate_flights() -> None:
    flight = ModelFlightRegistry.default().require("openai-api")

    with pytest.raises(ValueError, match="duplicate flight ids"):
        ModelFlightRegistry(flights=(flight, flight))


def test_catalog_signature_is_deterministic_for_same_models() -> None:
    catalog_a = ModelCatalog.default()
    catalog_b = ModelCatalog.default()
    assert catalog_a.signature() == catalog_b.signature()
    assert len(catalog_a.signature()) == 12


def test_quality_latency_and_speed_tier_boundaries() -> None:
    assert _quality_tier(0.6) == 1
    assert _quality_tier(1.8) == 2
    assert _quality_tier(4.0) == 3
    assert _quality_tier(7.0) == 4
    assert _quality_tier(14.0) == 5

    assert _latency_tier(0.6, "q4_k_m") == "fast"
    assert _latency_tier(4.0, "q8_0") == "medium"
    assert _latency_tier(7.0, "q6_k") == "medium"
    assert _latency_tier(14.0, "q4_k_m") == "slow"

    assert _speed_tier("fast", "q4_k_m") == 5
    assert _speed_tier("slow", "q8_0") == 1
    assert _speed_tier("medium", "q5_k_m") == 3


def test_memory_hint_is_positive_and_monotonic_with_size() -> None:
    small = _estimate_gguf_memory_hint(size_b=0.6, quant_bits=4)
    large = _estimate_gguf_memory_hint(size_b=14.0, quant_bits=4)
    higher_bits = _estimate_gguf_memory_hint(size_b=14.0, quant_bits=8)

    assert small.min_ram_gb >= 1.0
    assert small.min_vram_gb >= 0.5
    assert large.min_ram_gb > small.min_ram_gb
    assert large.min_vram_gb > small.min_vram_gb
    assert higher_bits.min_ram_gb > large.min_ram_gb
    assert higher_bits.min_vram_gb > large.min_vram_gb


def test_model_catalog_validation_and_lookup_error_paths() -> None:
    model = ModelFlightRegistry.default().require("openai-api").models[0]
    with pytest.raises(ValueError, match="description"):
        ModelFlight(flight_id="flight", description=" ", models=(model,))
    with pytest.raises(ValueError, match="at least one ModelFlight"):
        ModelFlightRegistry(flights=())

    first = ModelFlight(flight_id="first", description="First", models=(model,))
    second = ModelFlight(flight_id="second", description="Second", models=(model,))
    with pytest.raises(ValueError, match="duplicate model ids"):
        ModelFlightRegistry(flights=(first, second))
    with pytest.raises(ValueError, match="duplicate model ids"):
        ModelCatalog(models=(model, model))
    with pytest.raises(KeyError, match="Unknown model flight"):
        ModelFlightRegistry(flights=(first,)).require("missing")
    with pytest.raises(KeyError, match="Unknown model"):
        ModelCatalog(models=(model,)).require("missing")
    with pytest.raises(ValueError, match="repo_ids"):
        ModelCatalog.from_huggingface((" ",), api=object())


def test_model_catalog_filter_helper_rejects_each_mismatched_constraint() -> None:
    model = ModelSpec(
        model_id="local-model",
        provider="llama_cpp",
        family="family",
        size_b=3.0,
        format="gguf",
        quantization="q4_k_m",
        memory_hint=None,
        latency_hint=None,
        cost_hint=None,
        quality_tier=None,
        speed_tier=None,
        source="huggingface",
        capabilities=("chat",),
        tags=("local",),
    )
    defaults = {
        "provider": None,
        "family": None,
        "quantization": None,
        "model_format": None,
        "source": None,
        "local": None,
        "capability": None,
        "tag": None,
        "min_size_b": None,
        "max_size_b": None,
    }
    mismatches = {
        "provider": "openai",
        "family": "other",
        "quantization": "q8_0",
        "model_format": "api",
        "source": "curated",
        "local": False,
        "capability": "vision",
        "tag": "remote",
        "min_size_b": 4.0,
        "max_size_b": 2.0,
    }
    for key, value in mismatches.items():
        arguments = {**defaults, key: value}
        assert not catalog_impl._matches_catalog_filter(model, **arguments)


def test_huggingface_catalog_helpers_cover_malformed_and_inferred_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="callable model_info"):
        catalog_impl._call_huggingface_model_info(
            object(),
            repo_id="repo/model",
            revision=None,
            token=None,
            timeout=None,
        )

    real_import = builtins.__import__

    def _missing_hub(name: str, *args: object, **kwargs: object):
        if name == "huggingface_hub":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_hub)
    with pytest.raises(ImportError, match="huggingface_hub"):
        catalog_impl._load_huggingface_api(token=None)

    assert catalog_impl._huggingface_info_tags(SimpleNamespace(tags="invalid")) == ()
    assert catalog_impl._preferred_huggingface_artifact(SimpleNamespace(siblings=object())) is None
    assert catalog_impl._preferred_huggingface_artifact(SimpleNamespace(siblings=[])) is None
    assert catalog_impl._artifact_sort_key("model.gguf")[0] == 0
    assert catalog_impl._artifact_sort_key("model.safetensors")[0] == 1
    assert catalog_impl._artifact_sort_key("model.bin")[0] == 2
    assert catalog_impl._artifact_sort_key("README.md")[0] == 3
    assert catalog_impl._infer_model_format_from_artifact(None) is None
    assert catalog_impl._infer_model_format_from_artifact("model.safetensors") == "safetensors"
    assert catalog_impl._infer_model_format_from_artifact("model.bin") == "pytorch"
    assert catalog_impl._infer_model_format_from_artifact("README.md") is None
    assert catalog_impl._infer_quantization_from_labels(("MODEL-Q6_K",)) == "q6_k"
    assert catalog_impl._infer_quantization_from_labels(("plain",)) is None
    assert catalog_impl._infer_size_b_from_labels(("mix-8x7b",)) == 56.0
    assert catalog_impl._infer_size_b_from_labels(("plain",)) is None
    assert catalog_impl._infer_family_from_repo_id("org/-") == "-"
    assert catalog_impl._huggingface_license(SimpleNamespace(cardData=None)) is None
    assert catalog_impl._huggingface_license(SimpleNamespace(cardData={"license": 3})) is None
    assert catalog_impl._huggingface_context_window(SimpleNamespace(config=None)) is None
    assert catalog_impl._huggingface_context_window(SimpleNamespace(config={"n_positions": 0})) is None
    assert catalog_impl._card_data_mapping(SimpleNamespace(card_data={"license": "x"})) == {"license": "x"}
    assert catalog_impl._read_optional_str_attr(SimpleNamespace(value=" "), "value") is None
    assert catalog_impl._read_optional_str_attr(SimpleNamespace(value=3), "value") is None
    assert catalog_impl._normalized_labels((" a ", "a", "", 2)) == ("a", "2")
