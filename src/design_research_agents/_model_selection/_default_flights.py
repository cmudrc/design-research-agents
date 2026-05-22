"""Default curated model flights."""

from __future__ import annotations

from collections.abc import Sequence

from ._catalog import ModelFlight
from ._sota_flights import (
    build_agentic_coding_flight,
    build_frontier_moe_flight,
    build_open_reasoning_flight,
    build_vision_language_flight,
)
from ._types import (
    LatencyTier,
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSpec,
)

_DEFAULT_GGUF_QUANTIZATIONS: tuple[tuple[str, int], ...] = (
    ("q4_k_m", 4),
    ("q5_k_m", 5),
    ("q6_k", 6),
    ("q8_0", 8),
)


def build_default_flights() -> list[ModelFlight]:
    """Return default model flights for common local and hosted model families.

    Returns:
        List of default model flights.
    """
    return [
        _build_gguf_family_flight(
            flight_id="qwen3-gguf",
            description="Qwen3 instruction GGUF variants crossed by size and quantization.",
            family="qwen3",
            base_models=(
                ("qwen3-0.6b-instruct", 0.6),
                ("qwen3-1.8b-instruct", 1.8),
                ("qwen3-4b-instruct", 4.0),
                ("qwen3-7b-instruct", 7.0),
                ("qwen3-14b-instruct", 14.0),
                ("qwen3-32b-instruct", 32.0),
            ),
            tags=("local", "gguf", "qwen"),
        ),
        _build_gguf_family_flight(
            flight_id="gemma3-gguf",
            description="Gemma 3 instruction GGUF variants crossed by size and quantization.",
            family="gemma3",
            base_models=(
                ("gemma-3-1b-it", 1.0),
                ("gemma-3-4b-it", 4.0),
                ("gemma-3-12b-it", 12.0),
                ("gemma-3-27b-it", 27.0),
            ),
            tags=("local", "gguf", "gemma"),
        ),
        _build_gguf_family_flight(
            flight_id="llama-gguf",
            description="Llama instruction GGUF variants spanning small local and large comparison models.",
            family="llama",
            base_models=(
                ("llama-3.2-1b-instruct", 1.0),
                ("llama-3.2-3b-instruct", 3.0),
                ("llama-3.1-8b-instruct", 8.0),
                ("llama-3.1-70b-instruct", 70.0),
            ),
            tags=("local", "gguf", "llama"),
        ),
        _build_gguf_family_flight(
            flight_id="mistral-gguf",
            description="Mistral and Mixtral instruction GGUF variants for local and large-model comparisons.",
            family="mistral",
            base_models=(
                ("mistral-7b-instruct-v0.3", 7.0),
                ("mixtral-8x7b-instruct-v0.1", 46.7),
                ("mixtral-8x22b-instruct-v0.1", 141.0),
            ),
            tags=("local", "gguf", "mistral", "mixtral"),
        ),
        _build_gguf_family_flight(
            flight_id="phi-gguf",
            description="Phi instruction GGUF variants for compact local baselines.",
            family="phi",
            base_models=(
                ("phi-3.5-mini-instruct", 3.8),
                ("phi-4-mini-instruct", 3.8),
                ("phi-4", 14.0),
            ),
            tags=("local", "gguf", "phi"),
        ),
        build_open_reasoning_flight(),
        build_frontier_moe_flight(),
        build_agentic_coding_flight(),
        build_vision_language_flight(),
        _build_openai_flight(),
    ]


def _build_gguf_family_flight(
    *,
    flight_id: str,
    description: str,
    family: str,
    base_models: Sequence[tuple[str, float]],
    tags: Sequence[str],
) -> ModelFlight:
    """Build a GGUF model flight from size and quantization dimensions.

    Args:
        flight_id: Stable flight identifier.
        description: Human-readable flight description.
        family: Model family label to store on each model spec.
        base_models: Base model ids paired with approximate parameter counts in billions.
        tags: Discovery labels for the flight.

    Returns:
        Model flight containing every base-model/quantization combination.
    """
    models: list[ModelSpec] = []
    for base_name, size_b in base_models:
        quality_tier = _quality_tier(size_b)
        for quant_name, quant_bits in _DEFAULT_GGUF_QUANTIZATIONS:
            latency_tier = _latency_tier(size_b, quant_name)
            speed_tier = _speed_tier(latency_tier, quant_name)
            models.append(
                ModelSpec(
                    model_id=f"{base_name}-gguf-{quant_name}",
                    provider="llama_cpp",
                    family=family,
                    size_b=size_b,
                    format="gguf",
                    quantization=quant_name,
                    memory_hint=_estimate_gguf_memory_hint(size_b, quant_bits),
                    latency_hint=ModelLatencyHint(tier=latency_tier),
                    cost_hint=ModelCostHint(tier="low", usd_per_1k_tokens=0.0),
                    quality_tier=quality_tier,
                    speed_tier=speed_tier,
                    source="curated",
                    artifact=f"{base_name}-gguf-{quant_name}.gguf",
                    capabilities=("chat",),
                    tags=_normalized_labels((*tags, family, quant_name, f"{size_b:g}b")),
                )
            )
    return ModelFlight(
        flight_id=flight_id,
        description=description,
        models=tuple(models),
        tags=tuple(tags),
    )


def _build_openai_flight() -> ModelFlight:
    """Build the hosted OpenAI API model flight.

    Returns:
        Model flight containing hosted OpenAI API candidates.
    """
    return ModelFlight(
        flight_id="openai-api",
        description="Hosted OpenAI API candidates for remote fallback and service comparisons.",
        models=(
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
                source="curated",
                capabilities=("chat", "structured_output"),
                tags=("remote", "api", "openai"),
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
                source="curated",
                capabilities=("chat", "structured_output"),
                tags=("remote", "api", "openai"),
            ),
        ),
        tags=("remote", "api", "openai"),
    )


def _estimate_gguf_memory_hint(size_b: float, quant_bits: int) -> ModelMemoryHint:
    """Estimate GGUF memory needs from parameter count and quantization bits.

    Args:
        size_b: Approximate parameter count in billions.
        quant_bits: Quantization bit width.

    Returns:
        Approximate memory hint for local runtimes.
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
    """Return a rough quality tier from model size.

    Args:
        size_b: Approximate parameter count in billions.

    Returns:
        Quality tier from 1 to 5.
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
    """Return a rough latency tier from model size and quantization.

    Args:
        size_b: Approximate parameter count in billions.
        quant_name: Quantization label.

    Returns:
        Relative latency tier.
    """
    if size_b <= 1.0:
        return "fast"
    if size_b <= 4.0:
        return "fast" if quant_name in {"q4_k_m", "q5_k_m"} else "medium"
    if size_b <= 7.0:
        return "medium"
    return "slow"


def _speed_tier(latency_tier: LatencyTier, quant_name: str) -> int:
    """Return a rough speed tier from latency and quantization.

    Args:
        latency_tier: Relative latency tier.
        quant_name: Quantization label.

    Returns:
        Speed tier from 1 to 5.
    """
    base = {"fast": 5, "medium": 3, "slow": 1}[latency_tier]
    if quant_name == "q4_k_m":
        base += 1
    if quant_name == "q8_0":
        base -= 1
    return max(1, min(5, base))


def _normalized_labels(values: Sequence[object]) -> tuple[str, ...]:
    """Normalize and deduplicate labels.

    Args:
        values: Label-like values to normalize.

    Returns:
        Tuple of unique non-empty labels in first-observed order.
    """
    normalized: list[str] = []
    for value in values:
        label = str(value).strip()
        if label and label not in normalized:
            normalized.append(label)
    return tuple(normalized)
