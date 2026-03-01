from __future__ import annotations

from design_research_agents._model_selection._catalog import (
    ModelCatalog,
    _estimate_gguf_memory_hint,
    _latency_tier,
    _quality_tier,
    _speed_tier,
)


def test_default_catalog_has_expected_shape_and_is_discoverable() -> None:
    catalog = ModelCatalog.default()
    assert len(catalog.models) >= 20
    assert catalog.find("gpt-4o") is not None
    assert catalog.find("does-not-exist") is None


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
