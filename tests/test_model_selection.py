from __future__ import annotations

import pytest

from design_research_agents._model_selection import ModelSelector
from design_research_agents._model_selection._catalog import ModelCatalog
from design_research_agents._model_selection._hardware import HardwareProfile
from design_research_agents._model_selection._types import (
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSelectionDecision,
    ModelSpec,
)
from design_research_agents.llm import (
    OpenAICompatibleHTTPLLMClient,
    OpenAIServiceLLMClient,
    TransformersLocalLLMClient,
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
            ModelMemoryHint(min_ram_gb=min_ram_gb, min_vram_gb=None, note="test") if min_ram_gb is not None else None
        ),
        latency_hint=ModelLatencyHint(tier="medium"),
        cost_hint=ModelCostHint(tier="low", usd_per_1k_tokens=0.0),
        quality_tier=quality_tier,
        speed_tier=speed_tier,
    )


def _profile(
    *,
    available_ram_gb: float,
    cpu_count: int,
    load_average: tuple[float, float, float] = (0.2, 0.2, 0.2),
) -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=16.0,
        available_ram_gb=available_ram_gb,
        cpu_count=cpu_count,
        load_average=load_average,
        gpu_present=False,
        gpu_vram_gb=None,
        gpu_name=None,
        platform_name="test",
    )


def test_select_decision_prefers_local_when_fit() -> None:
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
    selector = ModelSelector(catalog=ModelCatalog(models=(local_small, local_large, remote)))

    decision = selector.select(
        task="summarize",
        priority="quality",
        hardware_profile=_profile(available_ram_gb=8.0, cpu_count=8),
        output="decision",
    )

    assert isinstance(decision, ModelSelectionDecision)
    assert decision.model_id == "local-large"


def test_select_decision_falls_back_to_remote_when_no_local_fit() -> None:
    local_model = _make_model(
        model_id="local-too-big",
        provider="llama_cpp",
        size_b=7.0,
        min_ram_gb=10.0,
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
    selector = ModelSelector(catalog=ModelCatalog(models=(local_model, remote_model)))

    decision = selector.select(
        task="chat",
        priority="balanced",
        max_cost_usd=0.05,
        max_latency_ms=1000,
        hardware_profile=_profile(available_ram_gb=1.0, cpu_count=4),
        output="decision",
    )

    assert isinstance(decision, ModelSelectionDecision)
    assert decision.model_id == "remote-fallback"
    assert decision.safety_constraints.max_latency_ms == 1000


def test_client_config_returns_required_shape_for_remote_provider() -> None:
    remote_model = _make_model(
        model_id="gpt-4o-mini",
        provider="openai",
        size_b=None,
        min_ram_gb=None,
        quality_tier=4,
        speed_tier=4,
    )
    selector = ModelSelector(catalog=ModelCatalog(models=(remote_model,)))

    config = selector.select(
        task="summarize",
        output="client_config",
    )

    assert isinstance(config, dict)
    assert config["provider"] == "openai"
    assert config["model_id"] == "gpt-4o-mini"
    assert config["client_class"] == "OpenAIServiceLLMClient"
    assert config["kwargs"] == {"default_model": "gpt-4o-mini"}
    assert isinstance(config["rationale"], str)
    assert isinstance(config["policy_id"], str)
    assert isinstance(config["catalog_signature"], str)


def test_client_output_instantiates_remote_client() -> None:
    remote_model = _make_model(
        model_id="gpt-4o-mini",
        provider="openai",
        size_b=None,
        min_ram_gb=None,
        quality_tier=4,
        speed_tier=4,
    )
    selector = ModelSelector(catalog=ModelCatalog(models=(remote_model,)))

    client = selector.select(task="summarize", output="client")

    assert isinstance(client, OpenAIServiceLLMClient)
    assert client.default_model() == "gpt-4o-mini"


def test_client_output_instantiates_transformers_client() -> None:
    local_model = _make_model(
        model_id="distilgpt2",
        provider="transformers_local",
        size_b=0.1,
        min_ram_gb=1.0,
        quality_tier=1,
        speed_tier=5,
    )
    selector = ModelSelector(catalog=ModelCatalog(models=(local_model,)))

    client = selector.select(task="summarize", output="client")

    assert isinstance(client, TransformersLocalLLMClient)
    assert client.default_model() == "distilgpt2"


def test_local_provider_requires_resolver_for_client_config_and_client() -> None:
    local_model = _make_model(
        model_id="qwen3-4b-instruct-gguf-q4_k_m",
        provider="llama_cpp",
        size_b=4.0,
        min_ram_gb=6.0,
        quality_tier=3,
        speed_tier=3,
    )
    selector = ModelSelector(catalog=ModelCatalog(models=(local_model,)))

    with pytest.raises(ValueError, match="local_client_resolver"):
        selector.select(task="summarize", output="client_config")

    with pytest.raises(ValueError, match="local_client_resolver"):
        selector.select(task="summarize", output="client")


def test_local_provider_with_resolver_returns_config_and_client() -> None:
    local_model = _make_model(
        model_id="qwen3-4b-instruct-gguf-q4_k_m",
        provider="llama_cpp",
        size_b=4.0,
        min_ram_gb=6.0,
        quality_tier=3,
        speed_tier=3,
    )
    selector = ModelSelector(
        catalog=ModelCatalog(models=(local_model,)),
        local_client_resolver=lambda decision: {
            "client_class": "OpenAICompatibleHTTPLLMClient",
            "kwargs": {"default_model": decision.model_id},
            "resolver_source": "custom",
        },
    )

    config = selector.select(task="summarize", output="client_config")
    assert isinstance(config, dict)
    assert config["client_class"] == "OpenAICompatibleHTTPLLMClient"
    assert config["kwargs"] == {"default_model": "qwen3-4b-instruct-gguf-q4_k_m"}
    assert config["resolver_source"] == "custom"

    client = selector.select(task="summarize", output="client")
    assert isinstance(client, OpenAICompatibleHTTPLLMClient)
    assert client.default_model() == "qwen3-4b-instruct-gguf-q4_k_m"


def test_hardware_profile_accepts_mapping_input() -> None:
    remote_model = _make_model(
        model_id="gpt-4o-mini",
        provider="openai",
        size_b=None,
        min_ram_gb=None,
        quality_tier=4,
        speed_tier=4,
    )
    selector = ModelSelector(catalog=ModelCatalog(models=(remote_model,)))

    decision = selector.select(
        task="summarize",
        hardware_profile={
            "total_ram_gb": 16.0,
            "available_ram_gb": 8.0,
            "cpu_count": 8,
            "load_average": [0.1, 0.2, 0.3],
            "gpu_present": False,
            "gpu_vram_gb": None,
            "gpu_name": None,
            "platform_name": "test",
        },
        output="decision",
    )

    assert isinstance(decision, ModelSelectionDecision)
    assert decision.model_id == "gpt-4o-mini"


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("total_ram_gb", object()),
        ("available_ram_gb", object()),
        ("cpu_count", object()),
        ("total_ram_gb", True),
        ("cpu_count", True),
    ),
)
def test_hardware_profile_mapping_rejects_invalid_numeric_values(field_name: str, value: object) -> None:
    remote_model = _make_model(
        model_id="gpt-4o-mini",
        provider="openai",
        size_b=None,
        min_ram_gb=None,
        quality_tier=4,
        speed_tier=4,
    )
    selector = ModelSelector(catalog=ModelCatalog(models=(remote_model,)))

    with pytest.raises(ValueError):
        selector.select(
            task="summarize",
            hardware_profile={field_name: value},
            output="decision",
        )
