import json

from design_research_agents.model_selection import (
    HardwareProfile,
    ModelCatalog,
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSelectionConstraints,
    ModelSelectionIntent,
    ModelSelectionPolicy,
    ModelSpec,
)


def _make_model(
    *,
    model_id: str,
    provider: str,
    size_b: float | None,
    min_ram_gb: float | None,
    quality_tier: int,
    speed_tier: int,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        provider=provider,
        family="test-family",
        size_b=size_b,
        format="gguf" if provider == "llama_cpp" else "api",
        quantization="q4_k_m" if provider == "llama_cpp" else None,
        memory_hint=(
            ModelMemoryHint(min_ram_gb=min_ram_gb, min_vram_gb=None, note="test")
            if min_ram_gb is not None
            else None
        ),
        latency_hint=ModelLatencyHint(tier="medium"),
        cost_hint=ModelCostHint(tier="low", usd_per_1k_tokens=0.0),
        quality_tier=quality_tier,
        speed_tier=speed_tier,
    )


def test_hardware_profile_asdict_and_str() -> None:
    profile = HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        cpu_count=8,
        load_average=(0.1, 0.2, 0.3),
        gpu_present=True,
        gpu_vram_gb=4.0,
        gpu_name="Test GPU",
        platform_name="TestOS",
    )
    payload = profile.asdict()
    assert payload["total_ram_gb"] == 16.0
    assert payload["gpu_name"] == "Test GPU"
    parsed = json.loads(str(profile))
    assert parsed["available_ram_gb"] == 8.0


def test_model_catalog_default_contains_qwen3_and_remote() -> None:
    catalog = ModelCatalog.default()
    assert any(model.family == "qwen3" for model in catalog.models)
    assert any(model.provider == "openai" for model in catalog.models)
    assert any(model.quantization == "q4_k_m" for model in catalog.models)


def test_model_selection_policy_prefers_local_when_fit() -> None:
    local_small = _make_model(
        model_id="local-small",
        provider="llama_cpp",
        size_b=1.0,
        min_ram_gb=2.0,
        quality_tier=1,
        speed_tier=4,
    )
    local_large = _make_model(
        model_id="local-large",
        provider="llama_cpp",
        size_b=4.0,
        min_ram_gb=6.0,
        quality_tier=3,
        speed_tier=3,
    )
    remote = _make_model(
        model_id="remote-best",
        provider="openai",
        size_b=None,
        min_ram_gb=None,
        quality_tier=5,
        speed_tier=4,
    )
    catalog = ModelCatalog(models=(local_small, local_large, remote))
    policy = ModelSelectionPolicy(catalog=catalog)
    decision = policy.select_model(
        intent=ModelSelectionIntent(task="summarize", priority="quality"),
        constraints=None,
        hardware_profile=HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            cpu_count=8,
            load_average=(0.2, 0.2, 0.2),
            gpu_present=False,
            gpu_vram_gb=None,
            gpu_name=None,
            platform_name="test",
        ),
    )
    assert decision.model_id == "local-large"


def test_model_selection_policy_falls_back_to_remote_when_no_local_fit() -> None:
    local_model = _make_model(
        model_id="local-too-big",
        provider="llama_cpp",
        size_b=7.0,
        min_ram_gb=8.0,
        quality_tier=4,
        speed_tier=2,
    )
    remote_model = _make_model(
        model_id="remote-fallback",
        provider="openai",
        size_b=None,
        min_ram_gb=None,
        quality_tier=4,
        speed_tier=3,
    )
    catalog = ModelCatalog(models=(local_model, remote_model))
    policy = ModelSelectionPolicy(catalog=catalog)
    constraints = ModelSelectionConstraints(max_cost_usd=0.05, max_latency_ms=1000)
    decision = policy.select_model(
        intent=ModelSelectionIntent(task="chat", priority="balanced"),
        constraints=constraints,
        hardware_profile=HardwareProfile(
            total_ram_gb=4.0,
            available_ram_gb=1.0,
            cpu_count=4,
            load_average=(0.1, 0.1, 0.1),
            gpu_present=False,
            gpu_vram_gb=None,
            gpu_name=None,
            platform_name="test",
        ),
    )
    assert decision.model_id == "remote-fallback"
    assert decision.safety_constraints.max_latency_ms == 1000
