from __future__ import annotations

import pytest

from design_research_agents.model_selection import ModelSelector
from design_research_agents.model_selection.catalog import ModelCatalog
from design_research_agents.model_selection.hardware import HardwareProfile
from design_research_agents.model_selection.policy import (
    ModelSelectionPolicy,
    _apply_cost_constraints,
    _apply_provider_constraints,
    _fits_ram_budget,
    _ram_budget_gb,
    _select_candidate_pool,
    _should_prefer_remote,
    _vram_budget_gb,
)
from design_research_agents.model_selection.selector import (
    _build_client_from_config,
    _coerce_hardware_profile,
    _coerce_load_average,
    _coerce_optional_bool,
    _coerce_optional_float,
    _coerce_optional_int,
    _coerce_optional_str,
)
from design_research_agents.model_selection.types import (
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSafetyConstraints,
    ModelSelectionConstraints,
    ModelSelectionDecision,
    ModelSelectionIntent,
    ModelSelectionPolicyConfig,
    ModelSpec,
)


def _model(
    *,
    model_id: str,
    provider: str,
    min_ram_gb: float | None = None,
    min_vram_gb: float | None = None,
    usd_per_1k_tokens: float | None = 0.1,
) -> ModelSpec:
    memory_hint = (
        ModelMemoryHint(min_ram_gb=min_ram_gb, min_vram_gb=min_vram_gb, note="test")
        if min_ram_gb is not None or min_vram_gb is not None
        else None
    )
    cost_hint = (
        ModelCostHint(tier="low", usd_per_1k_tokens=usd_per_1k_tokens)
        if usd_per_1k_tokens is not None
        else None
    )
    return ModelSpec(
        model_id=model_id,
        provider=provider,
        family="test",
        size_b=1.0,
        format="gguf" if provider == "llama_cpp" else "api",
        quantization="q4_k_m" if provider == "llama_cpp" else None,
        memory_hint=memory_hint,
        latency_hint=ModelLatencyHint(tier="fast"),
        cost_hint=cost_hint,
        quality_tier=3,
        speed_tier=3,
    )


def _hardware(
    *,
    total_ram_gb: float | None = 16.0,
    available_ram_gb: float | None = 8.0,
    cpu_count: int | None = 8,
    load_average: tuple[float, float, float] | None = (0.2, 0.2, 0.2),
    gpu_vram_gb: float | None = None,
) -> HardwareProfile:
    return HardwareProfile(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_count=cpu_count,
        load_average=load_average,
        gpu_present=False,
        gpu_vram_gb=gpu_vram_gb,
        gpu_name=None,
        platform_name="test",
    )


def test_policy_helper_edge_paths_cover_remaining_branches() -> None:
    config = ModelSelectionPolicyConfig(
        ram_reserve_gb=2.0,
        vram_reserve_gb=1.0,
        max_load_ratio=0.6,
        remote_cost_floor_usd=0.02,
    )
    total_only = _hardware(available_ram_gb=None, total_ram_gb=10.0)
    no_ram = _hardware(available_ram_gb=None, total_ram_gb=None)
    assert _ram_budget_gb(total_only, config) == 8.0
    assert _ram_budget_gb(no_ram, config) is None
    assert _vram_budget_gb(_hardware(gpu_vram_gb=0.5), config) == 0.0

    local = _model(model_id="local", provider="llama_cpp", min_ram_gb=6.0)
    remote = _model(model_id="remote", provider="openai", usd_per_1k_tokens=0.5)
    unknown_cost = _model(model_id="remote-unknown", provider="openai", usd_per_1k_tokens=None)

    assert (
        _fits_ram_budget(_model(model_id="no-hint", provider="llama_cpp"), ram_budget_gb=1.0)
        is True
    )
    assert _fits_ram_budget(local, ram_budget_gb=None) is True
    assert _should_prefer_remote(_hardware(load_average=None), config) is False

    constrained = _apply_provider_constraints(
        [local, remote],
        ModelSelectionConstraints(preferred_provider="openai"),
    )
    assert [model.model_id for model in constrained] == ["remote"]

    local_only = _apply_cost_constraints(
        [local, remote],
        ModelSelectionConstraints(max_cost_usd=0.0),
        remote_cost_floor_usd=config.remote_cost_floor_usd,
    )
    assert [model.model_id for model in local_only] == ["local"]
    assert _apply_cost_constraints(
        [unknown_cost],
        ModelSelectionConstraints(max_cost_usd=0.5),
        remote_cost_floor_usd=config.remote_cost_floor_usd,
    ) == [unknown_cost]

    intent = ModelSelectionIntent(task="test", priority="speed")
    pool, reason = _select_candidate_pool(
        intent=intent,
        constraints=ModelSelectionConstraints(require_local=True),
        prefer_remote_due_to_load=False,
        local_candidates=[local],
        remote_candidates=[remote],
        fallback_candidates=[remote],
    )
    assert pool == [local]
    assert reason == "local_required"

    pool, reason = _select_candidate_pool(
        intent=intent,
        constraints=ModelSelectionConstraints(require_local=True),
        prefer_remote_due_to_load=False,
        local_candidates=[],
        remote_candidates=[remote],
        fallback_candidates=[remote],
    )
    assert pool == [remote]
    assert reason == "local_required_no_fit"

    pool, reason = _select_candidate_pool(
        intent=intent,
        constraints=ModelSelectionConstraints(),
        prefer_remote_due_to_load=True,
        local_candidates=[local],
        remote_candidates=[remote],
        fallback_candidates=[local, remote],
    )
    assert pool == [remote]
    assert reason == "high_load_remote"

    pool, reason = _select_candidate_pool(
        intent=intent,
        constraints=ModelSelectionConstraints(),
        prefer_remote_due_to_load=False,
        local_candidates=[local],
        remote_candidates=[],
        fallback_candidates=[local],
    )
    assert pool == [local]
    assert reason == "local_only"


def test_policy_select_model_raises_for_empty_catalog_and_filtered_out_candidates() -> None:
    policy = ModelSelectionPolicy(catalog=ModelCatalog(models=()))
    with pytest.raises(ValueError, match="Model catalog is empty"):
        policy.select_model(
            intent=ModelSelectionIntent(task="summarize"),
            constraints=None,
            hardware_profile=_hardware(),
        )

    remote_only = _model(model_id="remote", provider="openai", usd_per_1k_tokens=0.5)
    policy_remote = ModelSelectionPolicy(catalog=ModelCatalog(models=(remote_only,)))
    with pytest.raises(ValueError, match="No model candidates available"):
        policy_remote.select_model(
            intent=ModelSelectionIntent(task="summarize"),
            constraints=ModelSelectionConstraints(max_cost_usd=0.0),
            hardware_profile=_hardware(),
        )


def test_selector_error_paths_and_alias_branches() -> None:
    selector = ModelSelector(
        catalog=ModelCatalog(models=(_model(model_id="gpt", provider="openai"),))
    )
    with pytest.raises(ValueError, match="output must be one of"):
        selector.select(task="summarize", output="invalid")  # type: ignore[arg-type]

    decision_http = ModelSelectionDecision(
        model_id="http-model",
        provider="openai-compatible-http",
        rationale="test",
        safety_constraints=ModelSafetyConstraints(max_cost_usd=None, max_latency_ms=None),
        policy_id="policy",
        catalog_signature="sig",
    )
    resolved_http = selector._resolve_client_config(decision_http)
    assert resolved_http["client_class"] == "OpenAICompatibleHTTPLLMClient"

    decision_mlx = ModelSelectionDecision(
        model_id="mlx-model",
        provider="mlx_local",
        rationale="test",
        safety_constraints=ModelSafetyConstraints(max_cost_usd=None, max_latency_ms=None),
        policy_id="policy",
        catalog_signature="sig",
    )
    resolved_mlx = selector._resolve_client_config(decision_mlx)
    assert resolved_mlx["client_class"] == "MlxLocalLLMClient"

    decision_vllm = ModelSelectionDecision(
        model_id="vllm-model",
        provider="vllm_local",
        rationale="test",
        safety_constraints=ModelSafetyConstraints(max_cost_usd=None, max_latency_ms=None),
        policy_id="policy",
        catalog_signature="sig",
    )
    resolved_vllm = selector._resolve_client_config(decision_vllm)
    assert resolved_vllm["client_class"] == "VllmServerLLMClient"

    decision_ollama = ModelSelectionDecision(
        model_id="ollama-model",
        provider="ollama_local",
        rationale="test",
        safety_constraints=ModelSafetyConstraints(max_cost_usd=None, max_latency_ms=None),
        policy_id="policy",
        catalog_signature="sig",
    )
    resolved_ollama = selector._resolve_client_config(decision_ollama)
    assert resolved_ollama["client_class"] == "OllamaLLMClient"

    decision_sglang = ModelSelectionDecision(
        model_id="sglang-model",
        provider="sglang_local",
        rationale="test",
        safety_constraints=ModelSafetyConstraints(max_cost_usd=None, max_latency_ms=None),
        policy_id="policy",
        catalog_signature="sig",
    )
    resolved_sglang = selector._resolve_client_config(decision_sglang)
    assert resolved_sglang["client_class"] == "SglangServerLLMClient"

    local_catalog = ModelCatalog(models=(_model(model_id="local", provider="llama_cpp"),))
    with pytest.raises(ValueError, match="unsupported client_class"):
        ModelSelector(
            catalog=local_catalog,
            local_client_resolver=lambda _decision: {"client_class": "Bad", "kwargs": {}},
        ).select(task="x", output="client_config")

    with pytest.raises(ValueError, match="invalid kwargs"):
        ModelSelector(
            catalog=local_catalog,
            local_client_resolver=lambda _decision: {
                "client_class": "OpenAIServiceLLMClient",
                "kwargs": [],
            },
        ).select(task="x", output="client_config")

    with pytest.raises(ValueError, match="must return a dict payload"):
        ModelSelector(
            catalog=local_catalog,
            local_client_resolver=lambda _decision: "bad",  # type: ignore[return-value]
        ).select(task="x", output="client_config")

    with pytest.raises(ValueError, match="must include 'client_class' and 'kwargs'"):
        ModelSelector(
            catalog=local_catalog,
            local_client_resolver=lambda _decision: {"client_class": "OpenAIServiceLLMClient"},
        ).select(task="x", output="client_config")

    with pytest.raises(ValueError, match="unsupported client_class"):
        _build_client_from_config({"client_class": "Nope", "kwargs": {}})
    with pytest.raises(ValueError, match="invalid kwargs"):
        _build_client_from_config({"client_class": "OpenAIServiceLLMClient", "kwargs": []})


def test_model_spec_is_local_includes_new_local_providers() -> None:
    assert _model(model_id="vllm", provider="vllm_local").is_local is True
    assert _model(model_id="ollama", provider="ollama_local").is_local is True
    assert _model(model_id="sglang", provider="sglang_local").is_local is True
    assert _model(model_id="openai", provider="openai").is_local is False


def test_selector_coercion_helper_error_paths() -> None:
    with pytest.raises(ValueError, match="mapping"):
        _coerce_hardware_profile(object())
    with pytest.raises(ValueError, match="3-item sequence"):
        _coerce_load_average([1, 2])
    with pytest.raises(ValueError, match="float-compatible"):
        _coerce_optional_float("not-a-float")
    with pytest.raises(ValueError, match="int-compatible"):
        _coerce_optional_int("not-an-int")
    with pytest.raises(ValueError, match="bool value"):
        _coerce_optional_bool("true")
    with pytest.raises(ValueError, match="str value"):
        _coerce_optional_str(123)
