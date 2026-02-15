"""Hugging Face Transformers backend wrapper for local text generation.

This module keeps the Transformers import lazy so other backends can be used
without installing optional dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransformersBackend:
    """Backend wrapper around ``transformers.pipeline`` for text generation."""

    model: str
    tokenizer: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    device: int | str | None = None
    device_map: str | None = None
    torch_dtype: object | None = None
    cache_dir: str | None = None
    use_fast: bool = True
    pipeline_task: str = "text-generation"
    model_kwargs: dict[str, object] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, object] = field(default_factory=dict)
    pipeline_kwargs: dict[str, object] = field(default_factory=dict)
    generation_kwargs: dict[str, object] = field(default_factory=dict)
    _pipeline: Any | None = field(default=None, init=False, repr=False)

    def close(self) -> None:
        """Release cached pipeline resources."""
        self._pipeline = None

    def complete(
        self,
        prompt: str,
        *,
        generation_kwargs: Mapping[str, object] | None = None,
    ) -> str:
        """Generate text using a Transformers pipeline."""
        pipeline = self._ensure_pipeline()
        merged_kwargs = dict(self.generation_kwargs)
        if generation_kwargs:
            merged_kwargs.update({key: value for key, value in generation_kwargs.items()})
        if self.pipeline_task == "text-generation":
            merged_kwargs.setdefault("return_full_text", False)
        output = pipeline(prompt, **merged_kwargs)
        text = _extract_text_from_pipeline(output)
        if not text:
            raise RuntimeError("Received an empty response from Transformers.")
        return text

    def _ensure_pipeline(self) -> Any:
        """Lazily build and cache the Transformers pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for backend='transformers'. "
                "Install with: pip install -e '.[local]'"
            ) from exc

        tokenizer_id = self.tokenizer or self.model
        tokenizer_kwargs = _merge_kwargs(
            self.tokenizer_kwargs,
            revision=self.revision,
            trust_remote_code=True if self.trust_remote_code else None,
            cache_dir=self.cache_dir,
        )
        tokenizer_kwargs.setdefault("use_fast", self.use_fast)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)

        model_kwargs = _merge_kwargs(
            self.model_kwargs,
            revision=self.revision,
            trust_remote_code=True if self.trust_remote_code else None,
            cache_dir=self.cache_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

        pipeline_kwargs = dict(self.pipeline_kwargs)
        if self.device is not None and "device" not in pipeline_kwargs:
            pipeline_kwargs["device"] = self.device

        self._pipeline = pipeline(
            self.pipeline_task,
            model=self.model,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs,
            **pipeline_kwargs,
        )
        return self._pipeline


def create_backend(
    model: str,
    *,
    tokenizer: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device: int | str | None = None,
    device_map: str | None = None,
    torch_dtype: object | None = None,
    cache_dir: str | None = None,
    use_fast: bool = True,
    pipeline_task: str = "text-generation",
    model_kwargs: Mapping[str, object] | None = None,
    tokenizer_kwargs: Mapping[str, object] | None = None,
    pipeline_kwargs: Mapping[str, object] | None = None,
    generation_kwargs: Mapping[str, object] | None = None,
) -> TransformersBackend:
    """Create a Transformers backend with normalized configuration."""
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model must not be empty.")

    normalized_tokenizer = tokenizer.strip() if tokenizer is not None else None
    if normalized_tokenizer == "":
        normalized_tokenizer = None

    normalized_revision = revision.strip() if revision is not None else None
    if normalized_revision == "":
        normalized_revision = None

    normalized_cache_dir = cache_dir.strip() if cache_dir is not None else None
    if normalized_cache_dir == "":
        normalized_cache_dir = None

    normalized_task = pipeline_task.strip()
    if not normalized_task:
        raise ValueError("pipeline_task must not be empty.")

    return TransformersBackend(
        model=normalized_model,
        tokenizer=normalized_tokenizer,
        revision=normalized_revision,
        trust_remote_code=trust_remote_code,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        cache_dir=normalized_cache_dir,
        use_fast=use_fast,
        pipeline_task=normalized_task,
        model_kwargs=dict(model_kwargs or {}),
        tokenizer_kwargs=dict(tokenizer_kwargs or {}),
        pipeline_kwargs=dict(pipeline_kwargs or {}),
        generation_kwargs=dict(generation_kwargs or {}),
    )


def _merge_kwargs(base: Mapping[str, object], **overrides: object) -> dict[str, object]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in merged:
            merged[key] = value
    return merged


def _extract_text_from_pipeline(output: Any) -> str:
    if isinstance(output, str):
        return output.strip()

    if isinstance(output, list):
        if not output:
            return ""
        return _extract_text_from_pipeline(output[0])

    if isinstance(output, dict):
        for key in ("generated_text", "summary_text", "translation_text", "text"):
            value = output.get(key)
            if isinstance(value, str):
                return value.strip()

    return ""
