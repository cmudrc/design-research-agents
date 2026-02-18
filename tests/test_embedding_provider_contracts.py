from __future__ import annotations

from types import SimpleNamespace

import pytest

from design_research_agents.contracts.llm import EmbeddingResult, LLMCapabilityError
from design_research_agents.memory.embedding import LLMEmbeddingProvider

pytestmark = pytest.mark.contract


def test_embedding_provider_model_name_resolution() -> None:
    provider = LLMEmbeddingProvider(llm_client=SimpleNamespace(default_model=lambda: "embed-model"))
    assert provider.model_name == "embed-model"

    fallback_default = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(default_model=lambda: (_ for _ in ()).throw(RuntimeError("x")))
    )
    assert fallback_default.model_name == "default"

    explicit = LLMEmbeddingProvider(llm_client=SimpleNamespace(), model="explicit-model")
    assert explicit.model_name == "explicit-model"


def test_embedding_provider_handles_missing_embed_callable() -> None:
    provider = LLMEmbeddingProvider(llm_client=SimpleNamespace(), enable_lexical_fallback=True)
    assert provider.embed(["hello"]) is None

    strict_provider = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(),
        enable_lexical_fallback=False,
    )
    with pytest.raises(RuntimeError, match="No embedding-capable callable"):
        strict_provider.embed(["hello"])


def test_embedding_provider_direct_embed_and_backend_embed_paths() -> None:
    direct = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(embed=lambda texts, model=None: [[len(texts), 1.5]]),
        model="m",
    )
    assert direct.embed(["a"]) == [[1.0, 1.5]]

    backend_client = SimpleNamespace(_backend=SimpleNamespace(embed=lambda texts: [[3, 4]]))
    via_backend = LLMEmbeddingProvider(llm_client=backend_client)
    assert via_backend.embed(["x"]) == [[3.0, 4.0]]


def test_embedding_provider_handles_capability_and_runtime_errors() -> None:
    provider = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(
            embed=lambda _texts: (_ for _ in ()).throw(LLMCapabilityError("nope"))
        ),
        enable_lexical_fallback=True,
    )
    assert provider.embed(["hello"]) is None

    strict_provider = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(embed=lambda _texts: (_ for _ in ()).throw(RuntimeError("bad"))),
        enable_lexical_fallback=False,
    )
    with pytest.raises(RuntimeError, match="bad"):
        strict_provider.embed(["hello"])


def test_embedding_provider_accepts_embedding_result_and_validates_counts() -> None:
    provider = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(
            embed=lambda texts: EmbeddingResult(vectors=[[1, 2] for _ in texts])
        ),
    )
    assert provider.embed(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]

    mismatch_fallback = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(embed=lambda _texts: [[1, 2]]),
        enable_lexical_fallback=True,
    )
    assert mismatch_fallback.embed(["a", "b"]) is None

    mismatch_strict = LLMEmbeddingProvider(
        llm_client=SimpleNamespace(embed=lambda _texts: [[1, 2]]),
        enable_lexical_fallback=False,
    )
    with pytest.raises(RuntimeError, match="mismatched vector count"):
        mismatch_strict.embed(["a", "b"])


def test_embedding_provider_empty_input_short_circuits() -> None:
    provider = LLMEmbeddingProvider(llm_client=SimpleNamespace(embed=lambda _texts: [[1, 2]]))
    assert provider.embed([]) == []
