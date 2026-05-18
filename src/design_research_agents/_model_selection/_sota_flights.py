"""Curated SOTA model flights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._catalog import ModelFlight
from ._types import (
    LatencyTier,
    ModelCostHint,
    ModelLatencyHint,
    ModelMemoryHint,
    ModelSpec,
)


def build_open_reasoning_flight() -> ModelFlight:
    """Build open-weight reasoning model references.

    Returns:
        Model flight containing reasoning-focused open-weight candidates.
    """
    return ModelFlight(
        flight_id="open-reasoning",
        description="Open-weight reasoning models for math, code, tool-use, and high-deliberation tasks.",
        models=(
            _reference_model(
                model_id="openai/gpt-oss-20b",
                provider="vllm_local",
                family="gpt-oss",
                size_b=21.0,
                model_format="mxfp4",
                quantization="mxfp4",
                min_ram_gb=16.0,
                min_vram_gb=16.0,
                memory_note="official_edge_target",
                latency_tier="medium",
                quality_tier=5,
                speed_tier=4,
                license_name="apache-2.0",
                context_window=128_000,
                capabilities=("chat", "reasoning", "tool_use", "coding", "long_context"),
                tags=("open-weight", "reasoning", "moe", "local", "vllm", "gpt-oss"),
                source_url="https://huggingface.co/openai/gpt-oss-20b",
                metadata={"active_parameters_b": 3.6, "source_note": "openai_gpt_oss_model_card"},
            ),
            _reference_model(
                model_id="openai/gpt-oss-120b",
                provider="vllm_local",
                family="gpt-oss",
                size_b=117.0,
                model_format="mxfp4",
                quantization="mxfp4",
                min_ram_gb=96.0,
                min_vram_gb=80.0,
                memory_note="single_80gb_gpu_target",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=2,
                license_name="apache-2.0",
                context_window=128_000,
                capabilities=("chat", "reasoning", "tool_use", "coding", "long_context"),
                tags=("open-weight", "reasoning", "moe", "local", "vllm", "gpt-oss"),
                source_url="https://huggingface.co/openai/gpt-oss-120b",
                metadata={"active_parameters_b": 5.1, "source_note": "openai_gpt_oss_model_card"},
            ),
            _reference_model(
                model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                provider="vllm_local",
                family="deepseek-r1",
                size_b=8.0,
                model_format="safetensors",
                min_ram_gb=18.0,
                min_vram_gb=16.0,
                memory_note="bf16_distill_estimate",
                latency_tier="medium",
                quality_tier=5,
                speed_tier=3,
                license_name="mit",
                capabilities=("chat", "reasoning", "coding"),
                tags=("open-weight", "reasoning", "distilled", "qwen", "vllm", "deepseek"),
                source_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-0528",
                metadata={"distilled_from": "deepseek-ai/DeepSeek-R1-0528"},
            ),
            _reference_model(
                model_id="deepseek-ai/DeepSeek-R1-0528",
                provider="vllm_local",
                family="deepseek-r1",
                size_b=671.0,
                model_format="safetensors",
                min_ram_gb=640.0,
                min_vram_gb=640.0,
                memory_note="frontier_moe_cluster_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=1,
                license_name="mit",
                capabilities=("chat", "reasoning", "coding", "tool_use"),
                tags=("open-weight", "reasoning", "moe", "vllm", "sglang", "deepseek"),
                source_url="https://huggingface.co/deepseek-ai/DeepSeek-R1-0528",
            ),
            _reference_model(
                model_id="microsoft/Phi-4-reasoning",
                provider="vllm_local",
                family="phi4",
                size_b=14.0,
                model_format="safetensors",
                min_ram_gb=30.0,
                min_vram_gb=28.0,
                memory_note="bf16_estimate",
                latency_tier="medium",
                quality_tier=4,
                speed_tier=3,
                license_name="mit",
                capabilities=("chat", "reasoning", "coding"),
                tags=("open-weight", "reasoning", "compact", "vllm", "sglang", "phi"),
                source_url="https://huggingface.co/microsoft/Phi-4-reasoning",
            ),
        ),
        tags=("open-weight", "reasoning"),
    )


def build_frontier_moe_flight() -> ModelFlight:
    """Build frontier-scale open-weight MoE model references.

    Returns:
        Model flight containing large MoE candidates for hosted or cluster-scale serving.
    """
    return ModelFlight(
        flight_id="frontier-moe-open-weights",
        description="Frontier-scale open-weight MoE models for long-context, agentic, and multimodal comparisons.",
        models=(
            _reference_model(
                model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
                provider="vllm_local",
                family="qwen3",
                size_b=235.0,
                model_format="safetensors",
                min_ram_gb=480.0,
                min_vram_gb=480.0,
                memory_note="bf16_cluster_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=1,
                license_name="apache-2.0",
                context_window=262_144,
                capabilities=("chat", "reasoning", "coding", "tool_use", "long_context", "multilingual"),
                tags=("open-weight", "frontier", "moe", "long-context", "qwen", "vllm", "sglang"),
                source_url="https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507",
                metadata={"active_parameters_b": 22.0},
            ),
            _reference_model(
                model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
                provider="vllm_local",
                family="qwen3-next",
                size_b=80.0,
                model_format="safetensors",
                min_ram_gb=160.0,
                min_vram_gb=160.0,
                memory_note="bf16_moe_estimate",
                latency_tier="medium",
                quality_tier=5,
                speed_tier=3,
                license_name="apache-2.0",
                context_window=262_144,
                capabilities=("chat", "coding", "long_context", "agentic"),
                tags=("open-weight", "efficient-moe", "long-context", "qwen", "vllm", "sglang"),
                source_url="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct",
                metadata={"active_parameters_b": 3.0},
            ),
            _reference_model(
                model_id="deepseek-ai/DeepSeek-V3.2",
                provider="vllm_local",
                family="deepseek-v3",
                size_b=685.0,
                model_format="safetensors",
                quantization="fp8_available",
                min_ram_gb=720.0,
                min_vram_gb=720.0,
                memory_note="frontier_moe_cluster_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=1,
                license_name="mit",
                capabilities=("chat", "reasoning", "coding", "tool_use", "agentic", "long_context"),
                tags=("open-weight", "frontier", "moe", "deepseek", "vllm", "sglang"),
                source_url="https://huggingface.co/deepseek-ai/DeepSeek-V3.2",
            ),
            _reference_model(
                model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                provider="vllm_local",
                family="llama4",
                size_b=109.0,
                model_format="safetensors",
                min_ram_gb=220.0,
                min_vram_gb=160.0,
                memory_note="moe_multimodal_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=2,
                license_name="llama4",
                context_window=10_000_000,
                capabilities=("chat", "vision", "multimodal", "long_context", "multilingual"),
                tags=("open-weight", "frontier", "moe", "vision", "llama", "vllm", "sglang"),
                source_url="https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct",
                metadata={"active_parameters_b": 17.0},
            ),
            _reference_model(
                model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                provider="vllm_local",
                family="llama4",
                size_b=400.0,
                model_format="safetensors",
                min_ram_gb=420.0,
                min_vram_gb=320.0,
                memory_note="fp8_single_h100_dgx_target",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=1,
                license_name="llama4",
                context_window=1_000_000,
                capabilities=("chat", "vision", "multimodal", "long_context", "multilingual"),
                tags=("open-weight", "frontier", "moe", "vision", "llama", "vllm", "sglang"),
                source_url="https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                metadata={"active_parameters_b": 17.0},
            ),
            _reference_model(
                model_id="zai-org/GLM-4.5",
                provider="vllm_local",
                family="glm4.5",
                size_b=355.0,
                model_format="safetensors",
                min_ram_gb=720.0,
                min_vram_gb=640.0,
                memory_note="frontier_moe_cluster_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=1,
                license_name=None,
                capabilities=("chat", "reasoning", "coding", "tool_use", "agentic"),
                tags=("open-weight", "frontier", "moe", "glm", "vllm", "agentic"),
                source_url="https://huggingface.co/zai-org/GLM-4.5",
                metadata={"active_parameters_b": 32.0},
            ),
            _reference_model(
                model_id="zai-org/GLM-4.5-Air",
                provider="vllm_local",
                family="glm4.5",
                size_b=106.0,
                model_format="safetensors",
                min_ram_gb=220.0,
                min_vram_gb=200.0,
                memory_note="compact_moe_cluster_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=2,
                license_name=None,
                capabilities=("chat", "reasoning", "coding", "tool_use", "agentic"),
                tags=("open-weight", "frontier", "compact-moe", "glm", "vllm", "agentic"),
                source_url="https://huggingface.co/zai-org/GLM-4.5-Air",
                metadata={"active_parameters_b": 12.0},
            ),
        ),
        tags=("open-weight", "frontier", "moe"),
    )


def build_agentic_coding_flight() -> ModelFlight:
    """Build agentic coding model references.

    Returns:
        Model flight containing coding and tool-use focused open-weight candidates.
    """
    return ModelFlight(
        flight_id="agentic-coding-open-weights",
        description="Open-weight models tuned for repository-scale coding, tool calls, and agent workflows.",
        models=(
            _reference_model(
                model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
                provider="vllm_local",
                family="qwen3-coder",
                size_b=480.0,
                model_format="safetensors",
                min_ram_gb=960.0,
                min_vram_gb=960.0,
                memory_note="frontier_coder_cluster_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=1,
                license_name="apache-2.0",
                context_window=256_000,
                capabilities=("chat", "coding", "tool_use", "agentic", "long_context"),
                tags=("open-weight", "coding", "agentic", "long-context", "qwen", "vllm", "sglang"),
                source_url="https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct",
                metadata={"active_parameters_b": 35.0, "extended_context_window": 1_000_000},
            ),
            _reference_model(
                model_id="moonshotai/Kimi-K2-Thinking",
                provider="vllm_local",
                family="kimi-k2",
                size_b=1_000.0,
                model_format="int4",
                quantization="native_int4",
                min_ram_gb=512.0,
                min_vram_gb=384.0,
                memory_note="native_int4_frontier_moe_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=2,
                license_name=None,
                context_window=256_000,
                capabilities=("chat", "reasoning", "coding", "tool_use", "agentic", "long_context"),
                tags=("open-weight", "coding", "agentic", "reasoning", "kimi", "moe", "vllm", "sglang"),
                source_url="https://huggingface.co/moonshotai/Kimi-K2-Thinking",
                metadata={"active_parameters_b": 32.0},
            ),
            _reference_model(
                model_id="MiniMaxAI/MiniMax-M2",
                provider="vllm_local",
                family="minimax-m2",
                size_b=229.0,
                model_format="safetensors",
                quantization="fp8_available",
                min_ram_gb=256.0,
                min_vram_gb=192.0,
                memory_note="agentic_moe_cluster_estimate",
                latency_tier="medium",
                quality_tier=5,
                speed_tier=3,
                license_name="mit",
                capabilities=("chat", "reasoning", "coding", "tool_use", "agentic"),
                tags=("open-weight", "coding", "agentic", "minimax", "moe", "vllm", "sglang"),
                source_url="https://huggingface.co/MiniMaxAI/MiniMax-M2",
            ),
        ),
        tags=("open-weight", "coding", "agentic"),
    )


def build_vision_language_flight() -> ModelFlight:
    """Build vision-language and multimodal model references.

    Returns:
        Model flight containing multimodal open-weight candidates.
    """
    return ModelFlight(
        flight_id="vision-language-open-weights",
        description="Open-weight VLMs for image understanding, document QA, and multimodal design artifacts.",
        models=(
            _reference_model(
                model_id="google/gemma-3-4b-it",
                provider="transformers_local",
                family="gemma3",
                size_b=4.0,
                model_format="safetensors",
                min_ram_gb=12.0,
                min_vram_gb=10.0,
                memory_note="bf16_vlm_estimate",
                latency_tier="medium",
                quality_tier=3,
                speed_tier=4,
                license_name="gemma",
                context_window=128_000,
                capabilities=("chat", "vision", "multimodal", "long_context"),
                tags=("open-weight", "vision", "multimodal", "gemma", "transformers"),
                source_url="https://huggingface.co/google/gemma-3-4b-it",
            ),
            _reference_model(
                model_id="google/gemma-3-12b-it",
                provider="transformers_local",
                family="gemma3",
                size_b=12.0,
                model_format="safetensors",
                min_ram_gb=28.0,
                min_vram_gb=24.0,
                memory_note="bf16_vlm_estimate",
                latency_tier="medium",
                quality_tier=4,
                speed_tier=3,
                license_name="gemma",
                context_window=128_000,
                capabilities=("chat", "vision", "multimodal", "long_context"),
                tags=("open-weight", "vision", "multimodal", "gemma", "transformers"),
                source_url="https://huggingface.co/google/gemma-3-12b-it",
            ),
            _reference_model(
                model_id="google/gemma-3-27b-it",
                provider="transformers_local",
                family="gemma3",
                size_b=27.0,
                model_format="safetensors",
                min_ram_gb=60.0,
                min_vram_gb=54.0,
                memory_note="bf16_vlm_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=2,
                license_name="gemma",
                context_window=128_000,
                capabilities=("chat", "vision", "multimodal", "long_context"),
                tags=("open-weight", "vision", "multimodal", "gemma", "transformers"),
                source_url="https://huggingface.co/google/gemma-3-27b-it",
            ),
            _reference_model(
                model_id="google/gemma-3n-E4B-it",
                provider="transformers_local",
                family="gemma3n",
                size_b=4.0,
                model_format="safetensors",
                min_ram_gb=10.0,
                min_vram_gb=8.0,
                memory_note="edge_multimodal_estimate",
                latency_tier="fast",
                quality_tier=3,
                speed_tier=5,
                license_name="gemma",
                capabilities=("chat", "vision", "audio", "multimodal", "edge"),
                tags=("open-weight", "vision", "audio", "edge", "gemma", "transformers"),
                source_url="https://huggingface.co/google/gemma-3n-E4B-it",
            ),
            _reference_model(
                model_id="Qwen/Qwen3-VL-32B-Instruct",
                provider="vllm_local",
                family="qwen3-vl",
                size_b=32.0,
                model_format="safetensors",
                min_ram_gb=72.0,
                min_vram_gb=64.0,
                memory_note="bf16_vlm_estimate",
                latency_tier="slow",
                quality_tier=5,
                speed_tier=2,
                license_name="apache-2.0",
                capabilities=("chat", "vision", "multimodal", "reasoning"),
                tags=("open-weight", "vision", "multimodal", "qwen", "vllm", "sglang"),
                source_url="https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct",
            ),
            _reference_model(
                model_id="microsoft/Phi-4-reasoning-vision-15B",
                provider="vllm_local",
                family="phi4",
                size_b=15.0,
                model_format="safetensors",
                min_ram_gb=34.0,
                min_vram_gb=30.0,
                memory_note="bf16_vlm_estimate",
                latency_tier="medium",
                quality_tier=4,
                speed_tier=3,
                license_name="mit",
                context_window=16_384,
                capabilities=("chat", "vision", "multimodal", "reasoning"),
                tags=("open-weight", "vision", "reasoning", "phi", "vllm"),
                source_url="https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B",
            ),
        ),
        tags=("open-weight", "vision", "multimodal"),
    )


def _reference_model(
    *,
    model_id: str,
    provider: str,
    family: str,
    size_b: float | None,
    model_format: str,
    quantization: str | None = None,
    min_ram_gb: float | None,
    min_vram_gb: float | None,
    memory_note: str,
    latency_tier: LatencyTier,
    quality_tier: int,
    speed_tier: int,
    license_name: str | None,
    context_window: int | None = None,
    capabilities: Sequence[str] = ("chat",),
    tags: Sequence[str] = (),
    source_url: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ModelSpec:
    """Build a curated non-GGUF reference model entry.

    Args:
        model_id: Upstream model id.
        provider: Runtime/provider key.
        family: Model family label.
        size_b: Approximate total parameter count in billions.
        model_format: Storage or runtime format label.
        quantization: Optional quantization label.
        min_ram_gb: Suggested minimum system RAM.
        min_vram_gb: Suggested minimum GPU VRAM.
        memory_note: Short memory assumption label.
        latency_tier: Relative latency tier.
        quality_tier: Relative quality tier.
        speed_tier: Relative speed tier.
        license_name: Upstream license label.
        context_window: Optional context-window size in tokens.
        capabilities: Capability labels.
        tags: Discovery labels.
        source_url: Optional upstream model-card URL.
        metadata: Optional supplemental metadata.

    Returns:
        Curated model spec.
    """
    return ModelSpec(
        model_id=model_id,
        provider=provider,
        family=family,
        size_b=size_b,
        format=model_format,
        quantization=quantization,
        memory_hint=ModelMemoryHint(min_ram_gb=min_ram_gb, min_vram_gb=min_vram_gb, note=memory_note),
        latency_hint=ModelLatencyHint(tier=latency_tier),
        cost_hint=ModelCostHint(tier="low", usd_per_1k_tokens=0.0),
        quality_tier=quality_tier,
        speed_tier=speed_tier,
        source="curated",
        repo_id=model_id,
        artifact=None,
        license=license_name,
        context_window=context_window,
        capabilities=_normalized_labels(capabilities),
        tags=_normalized_labels(tags),
        source_url=source_url or f"https://huggingface.co/{model_id}",
        metadata=dict(metadata or {}),
    )


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
