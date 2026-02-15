"""Shared model selection data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LatencyTier = Literal["fast", "medium", "slow"]
CostTier = Literal["low", "medium", "high"]
PriorityTier = Literal["quality", "balanced", "speed"]


@dataclass(slots=True, frozen=True)
class ModelMemoryHint:
    """Memory requirement hints for model selection.

    Attributes:
        min_ram_gb: Suggested minimum system RAM in GiB.
        min_vram_gb: Suggested minimum GPU VRAM in GiB.
        note: Optional annotation for the hint.
    """

    min_ram_gb: float | None
    min_vram_gb: float | None
    note: str | None = None


@dataclass(slots=True, frozen=True)
class ModelLatencyHint:
    """Latency hints for model selection.

    Attributes:
        tier: Qualitative latency tier.
        note: Optional annotation for the hint.
    """

    tier: LatencyTier
    note: str | None = None


@dataclass(slots=True, frozen=True)
class ModelCostHint:
    """Cost hints for model selection.

    Attributes:
        tier: Qualitative cost tier.
        usd_per_1k_tokens: Estimated cost per 1K tokens, when known.
    """

    tier: CostTier
    usd_per_1k_tokens: float | None = None


@dataclass(slots=True, frozen=True)
class ModelSpec:
    """Catalog entry describing one model option.

    Attributes:
        model_id: Unique model identifier used by backends.
        provider: Backend or provider name.
        family: Model family grouping label.
        size_b: Approximate parameter count in billions.
        format: Storage or API format identifier.
        quantization: Quantization name when applicable.
        memory_hint: Optional memory requirement hints.
        latency_hint: Optional latency hints.
        cost_hint: Optional cost hints.
        quality_tier: Relative quality score (higher is better).
        speed_tier: Relative speed score (higher is faster).
    """

    model_id: str
    provider: str
    family: str
    size_b: float | None
    format: str | None
    quantization: str | None
    memory_hint: ModelMemoryHint | None
    latency_hint: ModelLatencyHint | None
    cost_hint: ModelCostHint | None
    quality_tier: int | None
    speed_tier: int | None

    @property
    def is_local(self) -> bool:
        """Return True when the model runs locally.

        Returns:
            ``True`` when the provider is a local backend.
        """
        return self.provider in {"llama_cpp", "transformers_local", "mlx_local", "local"}


@dataclass(slots=True, frozen=True)
class ModelSelectionIntent:
    """Intent descriptor used by the model selection policy.

    Attributes:
        task: Description of the task or intent.
        priority: Preference for quality vs. speed.
    """

    task: str
    priority: PriorityTier = "balanced"

    def __post_init__(self) -> None:
        """Post-init validation."""
        normalized_task = self.task.strip()
        if not normalized_task:
            raise ValueError("intent.task must be non-empty.")
        if self.priority not in {"quality", "balanced", "speed"}:
            raise ValueError(f"Unsupported intent priority '{self.priority}'.")
        if normalized_task != self.task:
            object.__setattr__(self, "task", normalized_task)


@dataclass(slots=True, frozen=True)
class ModelSelectionConstraints:
    """Constraints that bound model selection choices.

    Attributes:
        require_local: Whether to force local-only selection.
        preferred_provider: Optional provider override.
        max_cost_usd: Optional maximum cost per 1K tokens.
        max_latency_ms: Optional latency cap in milliseconds.
    """

    require_local: bool = False
    preferred_provider: str | None = None
    max_cost_usd: float | None = None
    max_latency_ms: int | None = None

    def __post_init__(self) -> None:
        """Post-init validation."""
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0 when provided.")
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ValueError("max_latency_ms must be >= 0 when provided.")
        if self.preferred_provider is not None:
            normalized = self.preferred_provider.strip()
            object.__setattr__(self, "preferred_provider", normalized or None)


@dataclass(slots=True, frozen=True)
class ModelSafetyConstraints:
    """Safety bounds attached to a model selection decision.

    Attributes:
        max_cost_usd: Cost bound propagated into the decision.
        max_latency_ms: Latency bound propagated into the decision.
    """

    max_cost_usd: float | None
    max_latency_ms: int | None


@dataclass(slots=True, frozen=True)
class ModelSelectionDecision:
    """Selection output describing the chosen model and rationale.

    Attributes:
        model_id: Selected model identifier.
        provider: Selected provider name.
        rationale: Human-readable rationale for the choice.
        safety_constraints: Safety bounds applied to the selection.
        policy_id: Policy identifier for reproducibility.
        catalog_signature: Catalog signature used for the decision.
    """

    model_id: str
    provider: str
    rationale: str
    safety_constraints: ModelSafetyConstraints
    policy_id: str
    catalog_signature: str


@dataclass(slots=True, frozen=True)
class ModelSelectionPolicyConfig:
    """Configuration controlling model selection behavior.

    Attributes:
        policy_id: Identifier used for traceability.
        prefer_local: Whether to prefer local models by default.
        ram_reserve_gb: Reserved system RAM in GiB.
        vram_reserve_gb: Reserved GPU VRAM in GiB.
        max_load_ratio: Load ratio threshold to prefer remote.
        remote_cost_floor_usd: Cost below which remote is avoided.
        default_max_latency_ms: Default latency cap when none is provided.
    """

    policy_id: str = "default"
    prefer_local: bool = True
    ram_reserve_gb: float = 2.0
    vram_reserve_gb: float = 0.5
    max_load_ratio: float = 0.85
    remote_cost_floor_usd: float = 0.02
    default_max_latency_ms: int | None = None
