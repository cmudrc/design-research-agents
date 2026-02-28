"""Model catalog utilities and default catalog entries."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from ._types import (
    LatencyTier,
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSpec,
)


@dataclass(slots=True, frozen=True)
class ModelCatalog:
    """Catalog of known models and their hardware hints.

    Attributes:
        models: Tuple of model specifications.
    """

    models: tuple[ModelSpec, ...]
    """Stored ``models`` value."""

    @classmethod
    def default(cls) -> ModelCatalog:
        """Build the default model catalog.

        Returns:
            Default model catalog instance.
        """
        return cls(models=tuple(_build_default_models()))

    def signature(self) -> str:
        """Return a stable signature for catalog reproducibility.

        Returns:
            Stable signature string derived from the catalog contents.
        """
        payload = "|".join(
            f"{model.model_id}:{model.provider}:{model.quantization or ''}"
            for model in sorted(self.models, key=lambda item: item.model_id)
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:12]

    def find(self, model_id: str) -> ModelSpec | None:
        """Return the model spec with the given id, if present.

        Args:
            model_id: Model identifier to search for.

        Returns:
            Matching model spec, or ``None`` when not found.
        """
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None


def _build_default_models() -> list[ModelSpec]:
    """Return a list of default model specifications.

    Args:
        None.

    Returns:
        List of default model specifications.
    """
    # Seed the catalog with local Qwen3 GGUF variants and remote API models.
    models: list[ModelSpec] = []

    qwen3_sizes = [
        ("qwen3-0.6b-instruct", 0.6),
        ("qwen3-1.8b-instruct", 1.8),
        ("qwen3-4b-instruct", 4.0),
        ("qwen3-7b-instruct", 7.0),
        ("qwen3-14b-instruct", 14.0),
        ("qwen3-32b-instruct", 32.0),
    ]
    quantizations = [
        ("q4_k_m", 4),
        ("q5_k_m", 5),
        ("q6_k", 6),
        ("q8_0", 8),
    ]

    for base_name, size_b in qwen3_sizes:
        quality_tier = _quality_tier(size_b)
        for quant_name, quant_bits in quantizations:
            # Estimate memory and performance based on size and quantization.
            latency_tier = _latency_tier(size_b, quant_name)
            speed_tier = _speed_tier(latency_tier, quant_name)
            memory_hint = _estimate_gguf_memory_hint(size_b, quant_bits)
            models.append(
                ModelSpec(
                    model_id=f"{base_name}-gguf-{quant_name}",
                    provider="llama_cpp",
                    family="qwen3",
                    size_b=size_b,
                    format="gguf",
                    quantization=quant_name,
                    memory_hint=memory_hint,
                    latency_hint=ModelLatencyHint(tier=latency_tier),
                    cost_hint=ModelCostHint(tier="low", usd_per_1k_tokens=0.0),
                    quality_tier=quality_tier,
                    speed_tier=speed_tier,
                )
            )

    models.extend(
        [
            ModelSpec(
                model_id="gpt-4o-mini",
                provider="openai",
                family="gpt-4o",
                size_b=None,
                format="api",
                quantization=None,
                memory_hint=None,
                latency_hint=ModelLatencyHint(tier="medium"),
                cost_hint=ModelCostHint(tier="medium", usd_per_1k_tokens=0.01),
                quality_tier=3,
                speed_tier=4,
            ),
            ModelSpec(
                model_id="gpt-4o",
                provider="openai",
                family="gpt-4o",
                size_b=None,
                format="api",
                quantization=None,
                memory_hint=None,
                latency_hint=ModelLatencyHint(tier="slow"),
                cost_hint=ModelCostHint(tier="high", usd_per_1k_tokens=0.05),
                quality_tier=5,
                speed_tier=2,
            ),
        ]
    )
    return models


def _estimate_gguf_memory_hint(size_b: float, quant_bits: int) -> ModelMemoryHint:
    # Weight bytes are a rough proxy for runtime RAM/VRAM needs.
    """Estimate gguf memory hint.

    Args:
        size_b: Value supplied for ``size_b``.
        quant_bits: Value supplied for ``quant_bits``.

    Returns:
        Result produced by this call.
    """
    weight_bytes = size_b * 1e9 * (quant_bits / 8)
    weight_gb = weight_bytes / (1024**3)
    min_ram_gb = max(1.0, weight_gb * 1.35 + 0.6)
    min_vram_gb = max(0.5, weight_gb * 1.15 + 0.4)
    return ModelMemoryHint(
        min_ram_gb=round(min_ram_gb, 2),
        min_vram_gb=round(min_vram_gb, 2),
        note=f"estimate_{quant_bits}bit",
    )


def _quality_tier(size_b: float) -> int:
    """Quality tier.

    Args:
        size_b: Value supplied for ``size_b``.

    Returns:
        Result produced by this call.
    """
    if size_b <= 1.0:
        return 1
    if size_b <= 2.0:
        return 2
    if size_b <= 4.0:
        return 3
    if size_b <= 7.0:
        return 4
    return 5


def _latency_tier(size_b: float, quant_name: str) -> LatencyTier:
    """Latency tier.

    Args:
        size_b: Value supplied for ``size_b``.
        quant_name: Value supplied for ``quant_name``.

    Returns:
        Result produced by this call.
    """
    if size_b <= 1.0:
        return "fast"
    if size_b <= 4.0:
        return "fast" if quant_name in {"q4_k_m", "q5_k_m"} else "medium"
    if size_b <= 7.0:
        return "medium"
    return "slow"


def _speed_tier(latency_tier: LatencyTier, quant_name: str) -> int:
    """Speed tier.

    Args:
        latency_tier: Value supplied for ``latency_tier``.
        quant_name: Value supplied for ``quant_name``.

    Returns:
        Result produced by this call.
    """
    base = {"fast": 5, "medium": 3, "slow": 1}[latency_tier]
    if quant_name == "q4_k_m":
        base += 1
    if quant_name == "q8_0":
        base -= 1
    return max(1, min(5, base))
