"""Model catalog utilities and default catalog entries."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from ._types import (
    ModelCostHint,
    ModelSpec,
)

_SIZE_B_PATTERN = re.compile(r"(?:(?P<count>\d+)x)?(?P<size>\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)
_CONTEXT_WINDOW_KEYS = (
    "max_position_embeddings",
    "max_sequence_length",
    "n_positions",
    "seq_length",
    "sliding_window",
)


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelFlight:
    """Named, reproducible set of model candidates for experiments or selection.

    Attributes:
        flight_id: Stable identifier for the candidate set.
        description: Human-readable summary of the flight's purpose.
        models: Tuple of model specifications included in the flight.
        tags: Optional labels for discovery and grouping.
    """

    flight_id: str
    """Stable identifier for this flight."""
    description: str
    """Human-readable summary of this flight."""
    models: tuple[ModelSpec, ...]
    """Model specifications included in this flight."""
    tags: tuple[str, ...] = ()
    """Optional labels for discovery and grouping."""

    def __post_init__(self) -> None:
        """Validate and normalize flight fields."""
        normalized_flight_id = self.flight_id.strip()
        if not normalized_flight_id:
            raise ValueError("flight_id must be non-empty.")
        normalized_description = self.description.strip()
        if not normalized_description:
            raise ValueError("description must be non-empty.")
        normalized_models = tuple(self.models)
        if not normalized_models:
            raise ValueError("models must contain at least one ModelSpec.")
        duplicate_model_ids = _duplicate_model_ids(normalized_models)
        if duplicate_model_ids:
            duplicates = ", ".join(duplicate_model_ids)
            raise ValueError(f"ModelFlight contains duplicate model ids: {duplicates}.")
        normalized_tags = tuple(tag.strip() for tag in self.tags if tag.strip())
        object.__setattr__(self, "flight_id", normalized_flight_id)
        object.__setattr__(self, "description", normalized_description)
        object.__setattr__(self, "models", normalized_models)
        object.__setattr__(self, "tags", normalized_tags)

    def model_ids(self) -> tuple[str, ...]:
        """Return model ids in stable flight order.

        Returns:
            Tuple of model ids.
        """
        return tuple(model.model_id for model in self.models)

    def with_models(self, models: Sequence[ModelSpec]) -> ModelFlight:
        """Return a copy with a narrower model set and the same metadata.

        Args:
            models: Replacement model specs for the copied flight.

        Returns:
            New ``ModelFlight`` with the supplied models.
        """
        return ModelFlight(
            flight_id=self.flight_id,
            description=self.description,
            models=tuple(models),
            tags=self.tags,
        )


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelFlightRegistry:
    """Registry of named model flights.

    Attributes:
        flights: Tuple of known model flights.
    """

    flights: tuple[ModelFlight, ...]
    """Stored model flights."""

    def __post_init__(self) -> None:
        """Validate the catalog contains unique flight and model ids."""
        normalized_flights = tuple(self.flights)
        if not normalized_flights:
            raise ValueError("flights must contain at least one ModelFlight.")
        duplicate_flight_ids = _duplicate_flight_ids(normalized_flights)
        if duplicate_flight_ids:
            duplicates = ", ".join(duplicate_flight_ids)
            raise ValueError(f"ModelFlightRegistry contains duplicate flight ids: {duplicates}.")
        duplicate_model_ids = _duplicate_model_ids(self.collect_models(normalized_flights))
        if duplicate_model_ids:
            duplicates = ", ".join(duplicate_model_ids)
            raise ValueError(f"ModelFlightRegistry contains duplicate model ids: {duplicates}.")
        object.__setattr__(self, "flights", normalized_flights)

    @classmethod
    def default(cls) -> ModelFlightRegistry:
        """Build the default model-flight registry.

        Returns:
            Default model-flight registry instance.
        """
        from ._default_flights import build_default_flights

        return cls(flights=tuple(build_default_flights()))

    def find(self, flight_id: str) -> ModelFlight | None:
        """Return the flight with the given id, if present.

        Args:
            flight_id: Flight identifier to search for.

        Returns:
            Matching flight, or ``None`` when not found.
        """
        normalized_flight_id = flight_id.strip()
        for flight in self.flights:
            if flight.flight_id == normalized_flight_id:
                return flight
        return None

    def require(self, flight_id: str) -> ModelFlight:
        """Return the requested flight or raise a clear error.

        Args:
            flight_id: Flight identifier to retrieve.

        Returns:
            Matching flight.

        Raises:
            KeyError: If no flight exists for ``flight_id``.
        """
        flight = self.find(flight_id)
        if flight is None:
            available = ", ".join(self.flight_ids())
            raise KeyError(f"Unknown model flight '{flight_id}'. Available flights: {available}.")
        return flight

    def flight_ids(self) -> tuple[str, ...]:
        """Return all flight ids in stable catalog order.

        Returns:
            Tuple of flight ids.
        """
        return tuple(flight.flight_id for flight in self.flights)

    def collect_models(self, flights: Sequence[ModelFlight] | None = None) -> tuple[ModelSpec, ...]:
        """Return models from all or selected flights.

        Args:
            flights: Optional explicit flights to flatten. When omitted, all catalog flights
                are used.

        Returns:
            Flattened tuple of model specifications in flight order.
        """
        selected_flights = tuple(self.flights if flights is None else flights)
        return tuple(model for flight in selected_flights for model in flight.models)


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelCatalog:
    """Catalog of known models and their hardware hints.

    Attributes:
        models: Tuple of model specifications.
    """

    models: tuple[ModelSpec, ...]
    """Stored ``models`` value."""

    def __post_init__(self) -> None:
        """Validate and freeze model entries."""
        normalized_models = tuple(self.models)
        duplicate_model_ids = _duplicate_model_ids(normalized_models)
        if duplicate_model_ids:
            duplicates = ", ".join(duplicate_model_ids)
            raise ValueError(f"ModelCatalog contains duplicate model ids: {duplicates}.")
        object.__setattr__(self, "models", normalized_models)

    @classmethod
    def default(cls) -> ModelCatalog:
        """Build the default model catalog.

        Returns:
            Default model catalog instance.
        """
        return cls.from_flights(ModelFlightRegistry.default().flights)

    @classmethod
    def from_flights(cls, flights: Sequence[ModelFlight]) -> ModelCatalog:
        """Build a model catalog from one or more flights.

        Args:
            flights: Model flights to flatten into a model catalog.

        Returns:
            Model catalog containing every model from the supplied flights.
        """
        return cls(models=tuple(model for flight in flights for model in flight.models))

    @classmethod
    def from_huggingface(
        cls,
        repo_ids: Sequence[str],
        *,
        provider: str = "transformers_local",
        family: str | None = None,
        revision: str | None = None,
        model_format: str | None = None,
        token: bool | str | None = None,
        timeout: float | None = None,
        api: object | None = None,
        capabilities: Sequence[str] = ("chat",),
        tags: Sequence[str] = (),
        quality_tier: int | None = None,
        speed_tier: int | None = None,
    ) -> ModelCatalog:
        """Build a catalog from Hugging Face Hub model metadata.

        This method performs network I/O only when ``api`` is omitted. Tests and
        deterministic pipelines can pass a small object exposing ``model_info`` to
        avoid importing or calling ``huggingface_hub``.

        Args:
            repo_ids: Hugging Face repository ids to fetch.
            provider: Runtime/provider key to assign to discovered models.
            family: Optional family override for every discovered model.
            revision: Optional revision to request from the Hub.
            model_format: Optional format override for every discovered model.
            token: Optional Hugging Face token value passed to the Hub client.
            timeout: Optional request timeout.
            api: Optional preconfigured object exposing ``model_info``.
            capabilities: Capability labels to assign to discovered models.
            tags: Extra tags to assign to every discovered model.
            quality_tier: Optional quality score assigned to discovered models.
            speed_tier: Optional speed score assigned to discovered models.

        Returns:
            Catalog containing discovered Hugging Face models.

        Raises:
            ImportError: If ``huggingface_hub`` is needed but unavailable.
            ValueError: If no repository ids are provided.
        """
        normalized_repo_ids = tuple(repo_id.strip() for repo_id in repo_ids if repo_id.strip())
        if not normalized_repo_ids:
            raise ValueError("repo_ids must contain at least one non-empty repository id.")

        hf_api = api if api is not None else _load_huggingface_api(token=token)
        models: list[ModelSpec] = []
        for repo_id in normalized_repo_ids:
            info = _call_huggingface_model_info(
                hf_api,
                repo_id=repo_id,
                revision=revision,
                token=token,
                timeout=timeout,
            )
            models.append(
                _model_spec_from_huggingface_info(
                    info,
                    repo_id=repo_id,
                    provider=provider,
                    family=family,
                    revision=revision,
                    model_format=model_format,
                    capabilities=capabilities,
                    tags=tags,
                    quality_tier=quality_tier,
                    speed_tier=speed_tier,
                )
            )
        return cls(models=tuple(models))

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

    def model_ids(self) -> tuple[str, ...]:
        """Return all model ids in catalog order.

        Returns:
            Tuple of model ids.
        """
        return tuple(model.model_id for model in self.models)

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

    def require(self, model_id: str) -> ModelSpec:
        """Return a model by id or raise a clear error.

        Args:
            model_id: Model identifier to retrieve.

        Returns:
            Matching model spec.

        Raises:
            KeyError: If no model exists for ``model_id``.
        """
        model = self.find(model_id)
        if model is None:
            raise KeyError(f"Unknown model '{model_id}'.")
        return model

    def merge(
        self,
        *catalogs: ModelCatalog,
        replace: bool = False,
    ) -> ModelCatalog:
        """Merge this catalog with additional catalogs.

        Args:
            catalogs: Additional catalogs to append.
            replace: When true, later catalogs replace duplicate model ids. When false,
                duplicate model ids raise ``ValueError``.

        Returns:
            Merged model catalog.
        """
        merged: dict[str, ModelSpec] = {}
        model_ids: list[str] = []
        all_models = list(self.models)
        for catalog in catalogs:
            all_models.extend(catalog.models)

        for model in all_models:
            if model.model_id in merged and not replace:
                raise ValueError(f"Duplicate model id while merging catalogs: {model.model_id}.")
            if model.model_id not in merged:
                model_ids.append(model.model_id)
            merged[model.model_id] = model
        return ModelCatalog(models=tuple(merged[model_id] for model_id in model_ids))

    def filter(
        self,
        *,
        provider: str | None = None,
        family: str | None = None,
        quantization: str | None = None,
        model_format: str | None = None,
        source: str | None = None,
        local: bool | None = None,
        capability: str | None = None,
        tag: str | None = None,
        min_size_b: float | None = None,
        max_size_b: float | None = None,
    ) -> ModelCatalog:
        """Return a filtered catalog.

        Args:
            provider: Optional provider key to match.
            family: Optional model family to match.
            quantization: Optional quantization label to match.
            model_format: Optional model format to match.
            source: Optional provenance label to match.
            local: Optional local/remote filter.
            capability: Optional capability label to require.
            tag: Optional tag label to require.
            min_size_b: Optional minimum model size in billions.
            max_size_b: Optional maximum model size in billions.

        Returns:
            Catalog containing matching models in original order.
        """
        return ModelCatalog(
            models=tuple(
                model
                for model in self.models
                if _matches_catalog_filter(
                    model,
                    provider=provider,
                    family=family,
                    quantization=quantization,
                    model_format=model_format,
                    source=source,
                    local=local,
                    capability=capability,
                    tag=tag,
                    min_size_b=min_size_b,
                    max_size_b=max_size_b,
                )
            )
        )

    def by_provider(self, provider: str) -> ModelCatalog:
        """Return models for one provider."""
        return self.filter(provider=provider)

    def by_family(self, family: str) -> ModelCatalog:
        """Return models for one family."""
        return self.filter(family=family)

    def local(self) -> ModelCatalog:
        """Return local models."""
        return self.filter(local=True)

    def remote(self) -> ModelCatalog:
        """Return remote models."""
        return self.filter(local=False)

    def with_capability(self, capability: str) -> ModelCatalog:
        """Return models declaring one capability."""
        return self.filter(capability=capability)

    def with_tag(self, tag: str) -> ModelCatalog:
        """Return models declaring one tag."""
        return self.filter(tag=tag)


def _duplicate_flight_ids(flights: Sequence[ModelFlight]) -> tuple[str, ...]:
    """Return duplicated flight ids in first-observed order.

    Args:
        flights: Flights to inspect.

    Returns:
        Tuple of duplicated flight ids.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for flight in flights:
        if flight.flight_id in seen and flight.flight_id not in duplicates:
            duplicates.append(flight.flight_id)
        seen.add(flight.flight_id)
    return tuple(duplicates)


def _duplicate_model_ids(models: Sequence[ModelSpec]) -> tuple[str, ...]:
    """Return duplicated model ids in first-observed order.

    Args:
        models: Model specs to inspect.

    Returns:
        Tuple of duplicated model ids.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for model in models:
        if model.model_id in seen and model.model_id not in duplicates:
            duplicates.append(model.model_id)
        seen.add(model.model_id)
    return tuple(duplicates)


def _matches_catalog_filter(
    model: ModelSpec,
    *,
    provider: str | None,
    family: str | None,
    quantization: str | None,
    model_format: str | None,
    source: str | None,
    local: bool | None,
    capability: str | None,
    tag: str | None,
    min_size_b: float | None,
    max_size_b: float | None,
) -> bool:
    """Return whether one model matches catalog filter criteria."""
    if provider is not None and model.provider != provider:
        return False
    if family is not None and model.family != family:
        return False
    if quantization is not None and model.quantization != quantization:
        return False
    if model_format is not None and model.format != model_format:
        return False
    if source is not None and model.source != source:
        return False
    if local is not None and model.is_local is not local:
        return False
    if capability is not None and capability not in model.capabilities:
        return False
    if tag is not None and tag not in model.tags:
        return False
    if min_size_b is not None and (model.size_b is None or model.size_b < min_size_b):
        return False
    return max_size_b is None or (model.size_b is not None and model.size_b <= max_size_b)


def _load_huggingface_api(*, token: bool | str | None) -> object:
    """Load a Hugging Face API client lazily."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "ModelCatalog.from_huggingface() requires the optional 'huggingface_hub' package. "
            "Install design-research-agents with the 'huggingface' extra or install huggingface-hub directly."
        ) from exc
    return HfApi(token=token)


def _call_huggingface_model_info(
    api: object,
    *,
    repo_id: str,
    revision: str | None,
    token: bool | str | None,
    timeout: float | None,
) -> object:
    """Call ``model_info`` on a Hugging Face API-like object."""
    model_info = getattr(api, "model_info", None)
    if not callable(model_info):
        raise TypeError("api must expose callable model_info(...).")
    return model_info(
        repo_id=repo_id,
        revision=revision,
        timeout=timeout,
        files_metadata=True,
        token=token,
    )


def _model_spec_from_huggingface_info(
    info: object,
    *,
    repo_id: str,
    provider: str,
    family: str | None,
    revision: str | None,
    model_format: str | None,
    capabilities: Sequence[str],
    tags: Sequence[str],
    quality_tier: int | None,
    speed_tier: int | None,
) -> ModelSpec:
    """Build one model spec from Hugging Face ``ModelInfo``-like data."""
    hf_tags = _huggingface_info_tags(info)
    artifact = _preferred_huggingface_artifact(info)
    combined_tags = _normalized_labels((*tags, *hf_tags))
    resolved_revision = _read_optional_str_attr(info, "sha") or revision
    resolved_format = model_format or _infer_model_format_from_artifact(artifact)
    metadata = _huggingface_metadata(info)
    return ModelSpec(
        model_id=repo_id,
        provider=provider,
        family=family or _infer_family_from_repo_id(repo_id),
        size_b=_infer_size_b_from_labels((repo_id, *hf_tags)),
        format=resolved_format,
        quantization=_infer_quantization_from_labels((artifact or "", repo_id, *hf_tags)),
        memory_hint=None,
        latency_hint=None,
        cost_hint=ModelCostHint(tier="low", usd_per_1k_tokens=0.0) if _is_local_provider(provider) else None,
        quality_tier=quality_tier,
        speed_tier=speed_tier,
        source="huggingface",
        repo_id=repo_id,
        revision=resolved_revision,
        artifact=artifact,
        license=_huggingface_license(info),
        context_window=_huggingface_context_window(info),
        capabilities=_normalized_labels(capabilities),
        tags=combined_tags,
        source_url=f"https://huggingface.co/{repo_id}",
        metadata=metadata,
    )


def _is_local_provider(provider: str) -> bool:
    """Return whether a provider key is a local backend."""
    return provider in {
        "llama_cpp",
        "transformers_local",
        "mlx_local",
        "vllm_local",
        "ollama_local",
        "sglang_local",
        "local",
    }


def _huggingface_info_tags(info: object) -> tuple[str, ...]:
    """Return normalized tags from Hugging Face model info."""
    tags = getattr(info, "tags", ())
    if not isinstance(tags, Sequence) or isinstance(tags, str):
        return ()
    return _normalized_labels(str(tag) for tag in tags)


def _preferred_huggingface_artifact(info: object) -> str | None:
    """Return the preferred model artifact from Hugging Face siblings."""
    siblings = getattr(info, "siblings", ())
    if not isinstance(siblings, Sequence):
        return None
    filenames: list[str] = []
    for sibling in siblings:
        filename = _read_optional_str_attr(sibling, "rfilename")
        if filename is not None:
            filenames.append(filename)
    if not filenames:
        return None
    return sorted(filenames, key=_artifact_sort_key)[0]


def _artifact_sort_key(filename: str) -> tuple[int, str]:
    """Sort model artifact filenames by preferred runtime formats."""
    lower = filename.lower()
    if lower.endswith(".gguf"):
        return (0, filename)
    if lower.endswith(".safetensors"):
        return (1, filename)
    if lower.endswith(".bin"):
        return (2, filename)
    return (3, filename)


def _infer_model_format_from_artifact(artifact: str | None) -> str | None:
    """Infer model format from a filename."""
    if artifact is None:
        return None
    lower = artifact.lower()
    if lower.endswith(".gguf"):
        return "gguf"
    if lower.endswith(".safetensors"):
        return "safetensors"
    if lower.endswith(".bin"):
        return "pytorch"
    return None


def _infer_quantization_from_labels(labels: Sequence[str]) -> str | None:
    """Infer a quantization label from model identifiers, tags, or artifacts."""
    normalized = " ".join(label.lower() for label in labels)
    for quantization in ("q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "4bit", "8bit"):
        if quantization in normalized:
            return quantization
    return None


def _infer_size_b_from_labels(labels: Sequence[str]) -> float | None:
    """Infer parameter count in billions from labels."""
    for label in labels:
        match = _SIZE_B_PATTERN.search(label)
        if match is None:
            continue
        size = float(match.group("size"))
        count = match.group("count")
        if count is not None:
            size *= float(count)
        return size
    return None


def _infer_family_from_repo_id(repo_id: str) -> str:
    """Infer a compact family label from a Hugging Face repo id."""
    repo_name = repo_id.rsplit("/", maxsplit=1)[-1]
    family = repo_name.split("-", maxsplit=1)[0].strip().lower()
    return family or repo_name.lower()


def _huggingface_license(info: object) -> str | None:
    """Return a license label from Hugging Face model info when present."""
    card_data = _card_data_mapping(info)
    if card_data is None:
        return None
    raw_license = card_data.get("license")
    if isinstance(raw_license, str):
        return raw_license
    return None


def _huggingface_context_window(info: object) -> int | None:
    """Return a context window from Hugging Face config metadata when present."""
    config = getattr(info, "config", None)
    if not isinstance(config, Mapping):
        return None
    for key in _CONTEXT_WINDOW_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _huggingface_metadata(info: object) -> dict[str, object]:
    """Return stable supplemental metadata from Hugging Face model info."""
    metadata: dict[str, object] = {}
    for key in ("id", "author", "private", "gated", "downloads", "downloads_all_time", "likes"):
        value = getattr(info, key, None)
        if isinstance(value, str | int | bool):
            metadata[f"huggingface_{key}"] = value
    return metadata


def _card_data_mapping(info: object) -> Mapping[str, object] | None:
    """Return Hugging Face card data as a mapping when available."""
    for attr_name in ("cardData", "card_data"):
        value = getattr(info, attr_name, None)
        if isinstance(value, Mapping):
            return value
    return None


def _read_optional_str_attr(obj: object, attr_name: str) -> str | None:
    """Read one string attribute if present."""
    value = getattr(obj, attr_name, None)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _normalized_labels(values: Iterable[object]) -> tuple[str, ...]:
    """Normalize and deduplicate labels."""
    normalized: list[str] = []
    for value in values:
        label = str(value).strip()
        if label and label not in normalized:
            normalized.append(label)
    return tuple(normalized)
