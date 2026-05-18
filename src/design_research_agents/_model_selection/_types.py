"""Shared model selection data types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

type LatencyTier = Literal["fast", "medium", "slow"]
type CostTier = Literal["low", "medium", "high"]
type PriorityTier = Literal["quality", "balanced", "speed"]


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelMemoryHint:
    """Memory requirement hints for model selection.

    Attributes:
        min_ram_gb: Suggested minimum system RAM in GiB.
        min_vram_gb: Suggested minimum GPU VRAM in GiB.
        note: Optional annotation for the hint.
    """

    min_ram_gb: float | None
    """Suggested minimum system RAM (GiB) for reliable execution."""
    min_vram_gb: float | None
    """Suggested minimum GPU VRAM (GiB), when GPU execution is relevant."""
    note: str | None = None
    """Optional annotation explaining caveats in the memory hint."""


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelLatencyHint:
    """Latency hints for model selection."""

    tier: LatencyTier
    """Relative latency tier for this model option."""
    note: str | None = None
    """Optional annotation for latency assumptions."""


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelCostHint:
    """Cost hints for model selection."""

    tier: CostTier
    """Relative cost tier for this model option."""
    usd_per_1k_tokens: float | None = None
    """Estimated USD cost per 1K tokens, when available."""


@dataclass(slots=True, frozen=True, kw_only=True)
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
        source: Provenance label for the catalog entry.
        repo_id: Optional upstream repository id.
        revision: Optional upstream revision or commit SHA.
        artifact: Optional preferred model artifact or filename.
        license: Optional upstream license identifier.
        context_window: Optional context-window size in tokens.
        capabilities: Optional normalized capability labels.
        tags: Optional normalized discovery labels.
        source_url: Optional upstream URL.
        metadata: Optional supplemental metadata.
    """

    model_id: str
    """Provider-specific model identifier."""
    provider: str
    """Provider/backend key used to execute the model."""
    family: str
    """Model family grouping (for reporting and routing heuristics)."""
    size_b: float | None
    """Approximate parameter count in billions."""
    format: str | None
    """Model format identifier (for example GGUF or API-native)."""
    quantization: str | None
    """Quantization descriptor when applicable."""
    memory_hint: ModelMemoryHint | None
    """Optional memory requirements for this model."""
    latency_hint: ModelLatencyHint | None
    """Optional latency profile for this model."""
    cost_hint: ModelCostHint | None
    """Optional cost profile for this model."""
    quality_tier: int | None
    """Relative quality ranking used by policy scoring."""
    speed_tier: int | None
    """Relative speed ranking used by policy scoring."""
    source: str = "curated"
    """Provenance label for this catalog entry."""
    repo_id: str | None = None
    """Optional upstream repository id."""
    revision: str | None = None
    """Optional upstream revision or commit SHA."""
    artifact: str | None = None
    """Optional preferred model artifact or filename."""
    license: str | None = None
    """Optional upstream license identifier."""
    context_window: int | None = None
    """Optional context-window size in tokens."""
    capabilities: tuple[str, ...] = ()
    """Normalized capability labels for filtering and reporting."""
    tags: tuple[str, ...] = ()
    """Normalized discovery labels for filtering and reporting."""
    source_url: str | None = None
    """Optional upstream URL."""
    metadata: Mapping[str, object] | None = None
    """Optional supplemental metadata."""

    def __post_init__(self) -> None:
        """Normalize optional metadata fields for stable querying."""
        normalized_model_id = self.model_id.strip()
        if not normalized_model_id:
            raise ValueError("model_id must be non-empty.")
        normalized_provider = self.provider.strip()
        if not normalized_provider:
            raise ValueError("provider must be non-empty.")
        normalized_family = self.family.strip()
        if not normalized_family:
            raise ValueError("family must be non-empty.")
        normalized_source = self.source.strip() or "curated"
        normalized_capabilities = _normalize_string_tuple(self.capabilities)
        normalized_tags = _normalize_string_tuple(self.tags)
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError("context_window must be positive when provided.")

        object.__setattr__(self, "model_id", normalized_model_id)
        object.__setattr__(self, "provider", normalized_provider)
        object.__setattr__(self, "family", normalized_family)
        object.__setattr__(self, "source", normalized_source)
        object.__setattr__(self, "repo_id", _normalize_optional_string(self.repo_id))
        object.__setattr__(self, "revision", _normalize_optional_string(self.revision))
        object.__setattr__(self, "artifact", _normalize_optional_string(self.artifact))
        object.__setattr__(self, "license", _normalize_optional_string(self.license))
        object.__setattr__(self, "capabilities", normalized_capabilities)
        object.__setattr__(self, "tags", normalized_tags)
        object.__setattr__(self, "source_url", _normalize_optional_string(self.source_url))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def is_local(self) -> bool:
        """Return True when the model runs locally.

        Returns:
            ``True`` when the provider is a local backend.
        """
        # Keep this set aligned with provider keys emitted by catalog/selector layers.
        return self.provider in {
            "llama_cpp",
            "transformers_local",
            "mlx_local",
            "vllm_local",
            "ollama_local",
            "sglang_local",
            "local",
        }


def _normalize_optional_string(value: str | None) -> str | None:
    """Normalize optional string metadata."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_string_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and deduplicate a tuple of string labels."""
    normalized: list[str] = []
    for value in values:
        label = value.strip()
        if label and label not in normalized:
            normalized.append(label)
    return tuple(normalized)


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelSelectionIntent:
    """Intent descriptor used by the model selection policy."""

    task: str
    """Task description used to classify selection intent."""
    priority: PriorityTier = "balanced"
    """Priority tradeoff between quality and speed."""

    def __post_init__(self) -> None:
        """Post-init validation.

        Raises:
            ValueError: If task or priority values are invalid.
        """
        normalized_task = self.task.strip()
        if not normalized_task:
            raise ValueError("intent.task must be non-empty.")
        # Restrict priority to a closed set so scoring logic can remain branch-simple.
        if self.priority not in {"quality", "balanced", "speed"}:
            raise ValueError(f"Unsupported intent priority '{self.priority}'.")
        if normalized_task != self.task:
            object.__setattr__(self, "task", normalized_task)


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelSelectionConstraints:
    """Constraints that bound model selection choices.

    Attributes:
        require_local: Whether to force local-only selection.
        preferred_provider: Optional provider override.
        max_cost_usd: Optional maximum cost per 1K tokens.
        max_latency_ms: Optional latency cap in milliseconds.
    """

    require_local: bool = False
    """When true, only local providers are eligible."""
    preferred_provider: str | None = None
    """Optional preferred provider key to bias selection."""
    max_cost_usd: float | None = None
    """Optional maximum cost bound (USD per 1K tokens)."""
    max_latency_ms: int | None = None
    """Optional maximum latency bound in milliseconds."""

    def __post_init__(self) -> None:
        """Post-init validation.

        Raises:
            ValueError: If numeric constraints are negative.
        """
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0 when provided.")
        if self.max_latency_ms is not None and self.max_latency_ms < 0:
            raise ValueError("max_latency_ms must be >= 0 when provided.")
        if self.preferred_provider is not None:
            # Normalize provider hints early so policy checks don't need repeated strip logic.
            normalized = self.preferred_provider.strip()
            object.__setattr__(self, "preferred_provider", normalized or None)


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelSafetyConstraints:
    """Safety bounds attached to a model selection decision.

    Attributes:
        max_cost_usd: Cost bound propagated into the decision.
        max_latency_ms: Latency bound propagated into the decision.
    """

    max_cost_usd: float | None
    """Cost bound carried into the final decision payload."""
    max_latency_ms: int | None
    """Latency bound carried into the final decision payload."""


@dataclass(slots=True, frozen=True, kw_only=True)
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
    """Selected model identifier."""
    provider: str
    """Provider key for the selected model."""
    rationale: str
    """Human-readable explanation of the selection decision."""
    safety_constraints: ModelSafetyConstraints
    """Safety/cost/latency constraints attached to the decision."""
    policy_id: str
    """Policy identifier used to produce this decision."""
    catalog_signature: str
    """Catalog signature/version used during selection."""


@dataclass(slots=True, frozen=True, kw_only=True)
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
    """Policy identifier used for traceability."""
    prefer_local: bool = True
    """Whether local models are preferred by default."""
    ram_reserve_gb: float = 2.0
    """Reserved system RAM (GiB) not available to model workloads."""
    vram_reserve_gb: float = 0.5
    """Reserved GPU VRAM (GiB) not available to model workloads."""
    max_load_ratio: float = 0.85
    """System load threshold above which remote models are preferred."""
    remote_cost_floor_usd: float = 0.02
    """Remote cost floor below which remote options are deprioritized."""
    default_max_latency_ms: int | None = None
    """Default latency bound applied when callers provide none."""
