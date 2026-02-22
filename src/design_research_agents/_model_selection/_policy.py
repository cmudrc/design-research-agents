"""Model selection policy implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from design_research_agents._tracing import emit_model_selection_decision

from ._catalog import ModelCatalog
from ._hardware import HardwareProfile
from ._types import (
    ModelSafetyConstraints,
    ModelSelectionConstraints,
    ModelSelectionDecision,
    ModelSelectionIntent,
    ModelSelectionPolicyConfig,
    ModelSpec,
)


@dataclass(slots=True)
class ModelSelectionPolicy:
    """Policy that selects a model using intent, constraints, and hardware.

    Attributes:
        catalog: Model catalog used for candidate selection.
        config: Policy configuration values.
    """

    catalog: ModelCatalog = field(default_factory=ModelCatalog.default)
    """Field value for ``catalog``."""
    config: ModelSelectionPolicyConfig = field(default_factory=ModelSelectionPolicyConfig)
    """Field value for ``config``."""

    def select_model(
        self,
        *,
        intent: ModelSelectionIntent,
        constraints: ModelSelectionConstraints | None,
        hardware_profile: HardwareProfile | None,
    ) -> ModelSelectionDecision:
        """Select an appropriate model and emit a traceable decision.

        Args:
            intent: Task intent and priority preferences.
            constraints: Optional model selection constraints.
            hardware_profile: Optional hardware profile override.

        Returns:
            Selection decision with rationale and safety bounds.

        Raises:
            Exception: Raised when this operation cannot complete.
        """
        resolved_constraints = constraints or ModelSelectionConstraints()
        resolved_hardware = hardware_profile or HardwareProfile.detect()

        # Apply provider/cost filters before considering hardware fit.
        candidates = list(self.catalog.models)
        if not candidates:
            raise ValueError("Model catalog is empty.")
        candidates = _apply_provider_constraints(candidates, resolved_constraints)
        candidates = _apply_cost_constraints(
            candidates,
            resolved_constraints,
            remote_cost_floor_usd=self.config.remote_cost_floor_usd,
        )

        ram_budget_gb = _ram_budget_gb(resolved_hardware, self.config)
        vram_budget_gb = _vram_budget_gb(resolved_hardware, self.config)

        # Split candidates into local vs. remote for pool selection.
        local_candidates = [model for model in candidates if model.is_local and _fits_ram_budget(model, ram_budget_gb)]
        remote_candidates = [model for model in candidates if not model.is_local]

        prefer_remote_due_to_load = _should_prefer_remote(resolved_hardware, self.config)

        # Choose which candidate pool to score and then pick the best.
        selection_pool, selection_reason = _select_candidate_pool(
            intent=intent,
            constraints=resolved_constraints,
            prefer_remote_due_to_load=prefer_remote_due_to_load,
            local_candidates=local_candidates,
            remote_candidates=remote_candidates,
            fallback_candidates=candidates,
        )
        if not selection_pool:
            raise ValueError("No model candidates available after applying constraints.")

        selected_model = _pick_best_model(
            selection_pool,
            intent=intent,
            ram_budget_gb=ram_budget_gb,
            vram_budget_gb=vram_budget_gb,
            prefer_remote_due_to_load=prefer_remote_due_to_load,
        )

        # Attach safety bounds to the decision for downstream enforcement.
        safety_constraints = ModelSafetyConstraints(
            max_cost_usd=resolved_constraints.max_cost_usd,
            max_latency_ms=(
                resolved_constraints.max_latency_ms
                if resolved_constraints.max_latency_ms is not None
                else self.config.default_max_latency_ms
            ),
        )
        rationale = _build_rationale(
            selected_model=selected_model,
            intent=intent,
            constraints=resolved_constraints,
            hardware_profile=resolved_hardware,
            selection_reason=selection_reason,
            ram_budget_gb=ram_budget_gb,
            vram_budget_gb=vram_budget_gb,
        )
        decision = ModelSelectionDecision(
            model_id=selected_model.model_id,
            provider=selected_model.provider,
            rationale=rationale,
            safety_constraints=safety_constraints,
            policy_id=self.config.policy_id,
            catalog_signature=self.catalog.signature(),
        )

        # Emit a trace event to make the decision reproducible.
        emit_model_selection_decision(
            model_id=decision.model_id,
            provider=decision.provider,
            rationale=decision.rationale,
            policy_id=decision.policy_id,
            policy_config=self.config,
            catalog_signature=decision.catalog_signature,
            intent=intent,
            constraints=resolved_constraints,
            hardware_profile=resolved_hardware,
            candidate_count=len(selection_pool),
        )

        return decision


def _ram_budget_gb(
    hardware_profile: HardwareProfile,
    config: ModelSelectionPolicyConfig,
) -> float | None:
    """Run ram budget gb.

    Args:
        hardware_profile: Input value for this parameter.
        config: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if hardware_profile.available_ram_gb is not None:
        return max(0.0, hardware_profile.available_ram_gb - config.ram_reserve_gb)
    if hardware_profile.total_ram_gb is not None:
        return max(0.0, hardware_profile.total_ram_gb - config.ram_reserve_gb)
    return None


def _vram_budget_gb(
    hardware_profile: HardwareProfile,
    config: ModelSelectionPolicyConfig,
) -> float | None:
    """Run vram budget gb.

    Args:
        hardware_profile: Input value for this parameter.
        config: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if hardware_profile.gpu_vram_gb is None:
        return None
    return max(0.0, hardware_profile.gpu_vram_gb - config.vram_reserve_gb)


def _fits_ram_budget(model: ModelSpec, ram_budget_gb: float | None) -> bool:
    """Run fits ram budget.

    Args:
        model: Input value for this parameter.
        ram_budget_gb: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if model.memory_hint is None or model.memory_hint.min_ram_gb is None:
        return True
    if ram_budget_gb is None:
        return True
    return model.memory_hint.min_ram_gb <= ram_budget_gb


def _should_prefer_remote(hardware_profile: HardwareProfile, config: ModelSelectionPolicyConfig) -> bool:
    """Run should prefer remote.

    Args:
        hardware_profile: Input value for this parameter.
        config: Input value for this parameter.

    Returns:
        Computed return value.
    """
    load = hardware_profile.load_average
    cpu_count = hardware_profile.cpu_count
    if load is None or cpu_count is None or cpu_count <= 0:
        return False
    load_ratio = load[0] / cpu_count
    return load_ratio >= config.max_load_ratio


def _apply_provider_constraints(
    candidates: list[ModelSpec],
    constraints: ModelSelectionConstraints,
) -> list[ModelSpec]:
    """Run apply provider constraints.

    Args:
        candidates: Input value for this parameter.
        constraints: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if constraints.preferred_provider:
        preferred = [model for model in candidates if model.provider == constraints.preferred_provider]
        if preferred:
            return preferred
    return candidates


def _apply_cost_constraints(
    candidates: list[ModelSpec],
    constraints: ModelSelectionConstraints,
    *,
    remote_cost_floor_usd: float,
) -> list[ModelSpec]:
    """Run apply cost constraints.

    Args:
        candidates: Input value for this parameter.
        constraints: Input value for this parameter.
        remote_cost_floor_usd: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if constraints.max_cost_usd is None:
        return candidates
    if constraints.max_cost_usd <= remote_cost_floor_usd:
        return [model for model in candidates if model.is_local]
    filtered: list[ModelSpec] = []
    for model in candidates:
        if model.cost_hint is None or model.cost_hint.usd_per_1k_tokens is None:
            filtered.append(model)
            continue
        if model.cost_hint.usd_per_1k_tokens <= constraints.max_cost_usd:
            filtered.append(model)
    return filtered


def _select_candidate_pool(
    *,
    intent: ModelSelectionIntent,
    constraints: ModelSelectionConstraints,
    prefer_remote_due_to_load: bool,
    local_candidates: list[ModelSpec],
    remote_candidates: list[ModelSpec],
    fallback_candidates: list[ModelSpec],
) -> tuple[list[ModelSpec], str]:
    """Run select candidate pool.

    Args:
        intent: Input value for this parameter.
        constraints: Input value for this parameter.
        prefer_remote_due_to_load: Input value for this parameter.
        local_candidates: Input value for this parameter.
        remote_candidates: Input value for this parameter.
        fallback_candidates: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if constraints.require_local:
        if local_candidates:
            return local_candidates, "local_required"
        return fallback_candidates, "local_required_no_fit"
    if prefer_remote_due_to_load and remote_candidates:
        return remote_candidates, "high_load_remote"
    if local_candidates and (intent.priority != "speed"):
        return local_candidates, "local_fit"
    if local_candidates and not remote_candidates:
        return local_candidates, "local_only"
    if remote_candidates:
        return remote_candidates, "remote_fallback"
    if local_candidates:
        return local_candidates, "local_fallback"
    return fallback_candidates, "catalog_fallback"


def _pick_best_model(
    candidates: list[ModelSpec],
    *,
    intent: ModelSelectionIntent,
    ram_budget_gb: float | None,
    vram_budget_gb: float | None,
    prefer_remote_due_to_load: bool,
) -> ModelSpec:
    """Run pick best model.

    Args:
        candidates: Input value for this parameter.
        intent: Input value for this parameter.
        ram_budget_gb: Input value for this parameter.
        vram_budget_gb: Input value for this parameter.
        prefer_remote_due_to_load: Input value for this parameter.

    Returns:
        Computed return value.
    """
    scored = []
    for model in candidates:
        score = _score_model(
            model,
            intent=intent,
            ram_budget_gb=ram_budget_gb,
            vram_budget_gb=vram_budget_gb,
            prefer_remote_due_to_load=prefer_remote_due_to_load,
        )
        scored.append((score, model))
    size_direction = -1.0 if intent.priority == "speed" else 1.0
    scored.sort(
        key=lambda item: (
            item[0],
            (item[1].size_b or 0.0) * size_direction,
            item[1].model_id,
        ),
        reverse=True,
    )
    return scored[0][1]


def _score_model(
    model: ModelSpec,
    *,
    intent: ModelSelectionIntent,
    ram_budget_gb: float | None,
    vram_budget_gb: float | None,
    prefer_remote_due_to_load: bool,
) -> float:
    """Run score model.

    Args:
        model: Input value for this parameter.
        intent: Input value for this parameter.
        ram_budget_gb: Input value for this parameter.
        vram_budget_gb: Input value for this parameter.
        prefer_remote_due_to_load: Input value for this parameter.

    Returns:
        Computed return value.
    """
    quality = model.quality_tier or 0
    speed = model.speed_tier or 0
    if intent.priority == "quality":
        score = quality * 10 + speed
    elif intent.priority == "speed":
        score = speed * 10 + quality
    else:
        score = quality * 6 + speed * 4

    if model.is_local and prefer_remote_due_to_load:
        score -= 5

    if model.memory_hint and ram_budget_gb is not None:
        min_ram = model.memory_hint.min_ram_gb
        if min_ram is not None:
            headroom = ram_budget_gb - min_ram
            if headroom < 1:
                score -= 3
            elif headroom < 2:
                score -= 1

    if model.memory_hint and vram_budget_gb is not None:
        min_vram = model.memory_hint.min_vram_gb
        if min_vram is not None and min_vram > vram_budget_gb:
            if intent.priority == "speed":
                score -= 4
            else:
                score -= 1

    return score


def _build_rationale(
    *,
    selected_model: ModelSpec,
    intent: ModelSelectionIntent,
    constraints: ModelSelectionConstraints,
    hardware_profile: HardwareProfile,
    selection_reason: str,
    ram_budget_gb: float | None,
    vram_budget_gb: float | None,
) -> str:
    """Run build rationale.

    Args:
        selected_model: Input value for this parameter.
        intent: Input value for this parameter.
        constraints: Input value for this parameter.
        hardware_profile: Input value for this parameter.
        selection_reason: Input value for this parameter.
        ram_budget_gb: Input value for this parameter.
        vram_budget_gb: Input value for this parameter.

    Returns:
        Computed return value.
    """
    parts = [
        f"priority={intent.priority}",
        f"selection_reason={selection_reason}",
        (f"model_size_b={selected_model.size_b}" if selected_model.size_b is not None else None),
    ]
    if ram_budget_gb is not None:
        parts.append(f"ram_budget_gb={ram_budget_gb:.1f}")
    if vram_budget_gb is not None:
        parts.append(f"vram_budget_gb={vram_budget_gb:.1f}")
    if constraints.max_latency_ms is not None:
        parts.append(f"max_latency_ms={constraints.max_latency_ms}")
    if constraints.max_cost_usd is not None:
        parts.append(f"max_cost_usd={constraints.max_cost_usd}")
    if hardware_profile.gpu_present is None:
        parts.append("gpu_present=unknown")
    elif hardware_profile.gpu_present:
        parts.append("gpu_present=true")
    else:
        parts.append("gpu_present=false")
    return "; ".join(part for part in parts if part is not None)
