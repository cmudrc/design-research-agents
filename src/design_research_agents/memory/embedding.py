"""Embedding provider abstractions for memory retrieval backends."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Protocol, cast, runtime_checkable

from design_research_agents.contracts.llm import EmbeddingResult, LLMCapabilityError


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for converting text payloads into dense vectors."""

    @property
    def model_name(self) -> str:
        """Return stable embedding model identifier.

        Returns:
            Stable embedding model identifier.
        """

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Embed one or more texts.

        Args:
            texts: Input texts to embed.

        Returns:
            Embedding vectors in input order, or ``None`` when embeddings are
            unavailable and lexical fallback should be used.
        """


class LLMEmbeddingProvider:
    """Embedding adapter that delegates to an LLM client/backend when available."""

    def __init__(
        self,
        *,
        llm_client: object,
        model: str | None = None,
        enable_lexical_fallback: bool = True,
    ) -> None:
        """Initialize one embedding adapter from an existing LLM client.

        Args:
            llm_client: LLM client instance that may expose embeddings.
            model: Optional explicit embedding model identifier.
            enable_lexical_fallback: Return ``None`` instead of raising when
                embeddings are unsupported.
        """
        self._llm_client = llm_client
        self._enable_lexical_fallback = enable_lexical_fallback
        resolved_model_name = (model or "").strip()
        if not resolved_model_name:
            default_model_callable = getattr(llm_client, "default_model", None)
            if callable(default_model_callable):
                try:
                    resolved_model_name = str(default_model_callable()).strip()
                except Exception:
                    resolved_model_name = "default"
            else:
                resolved_model_name = "default"
        self._model_name = resolved_model_name

    @property
    def model_name(self) -> str:
        """Return configured embedding model identifier.

        Returns:
            Resolved embedding model identifier.
        """
        return self._model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Embed texts through available client/backend embedding APIs.

        Args:
            texts: Input texts to embed.

        Returns:
            List of vectors, or ``None`` when lexical fallback is enabled and
            embeddings are unavailable.

        Raises:
            Exception: Raised when embeddings fail and lexical fallback is
                disabled.
        """
        if not texts:
            return []

        embed_callable = self._resolve_embed_callable()
        if embed_callable is None:
            if self._enable_lexical_fallback:
                return None
            raise RuntimeError("No embedding-capable callable found on llm_client/backend.")

        try:
            result = self._invoke_embed(embed_callable, texts)
        except LLMCapabilityError:
            if self._enable_lexical_fallback:
                return None
            raise
        except Exception:
            if self._enable_lexical_fallback:
                return None
            raise

        vectors = result.vectors if isinstance(result, EmbeddingResult) else result

        normalized_vectors: list[list[float]] = []
        for vector in vectors:
            normalized_vectors.append([float(value) for value in vector])
        if len(normalized_vectors) != len(texts):
            if self._enable_lexical_fallback:
                return None
            raise RuntimeError("Embedding provider returned mismatched vector count.")
        return normalized_vectors

    def _resolve_embed_callable(
        self,
    ) -> Callable[..., EmbeddingResult | Sequence[Sequence[float]]] | None:
        """Return first available embedding callable from client or backend.

        Returns:
            Embed callable when available, otherwise ``None``.
        """
        direct_embed = getattr(self._llm_client, "embed", None)
        if callable(direct_embed):
            return cast(
                Callable[..., EmbeddingResult | Sequence[Sequence[float]]],
                direct_embed,
            )

        backend = getattr(self._llm_client, "_backend", None)
        backend_embed = getattr(backend, "embed", None)
        if callable(backend_embed):
            return cast(
                Callable[..., EmbeddingResult | Sequence[Sequence[float]]],
                backend_embed,
            )
        return None

    def _invoke_embed(
        self,
        embed_callable: Callable[..., EmbeddingResult | Sequence[Sequence[float]]],
        texts: Sequence[str],
    ) -> EmbeddingResult | Sequence[Sequence[float]]:
        """Invoke embed callable while adapting signature variants.

        Args:
            embed_callable: Resolved embedding callable.
            texts: Texts to embed.

        Returns:
            Raw embedding result from the resolved callable.
        """
        signature = inspect.signature(embed_callable)
        parameters = signature.parameters

        if "model" in parameters:
            return embed_callable(texts, model=self._model_name)
        return embed_callable(texts)


__all__ = ["EmbeddingProvider", "LLMEmbeddingProvider"]
