"""Zero-dependency HTML stand-in client implementation."""

from __future__ import annotations

from .._backends._providers._html import HTMLBackend
from ._shared import _config_hash, _SingleBackendLLMClient


class HTMLLLMClient(_SingleBackendLLMClient):
    """Deterministic HTML-wrapping client for offline demos and tracing."""

    def __init__(
        self,
        *,
        name: str = "html-model",
        default_model: str = "html-standin-v1",
    ) -> None:
        """Initialize the HTML stand-in client."""
        backend = HTMLBackend(
            name=name,
            model=default_model,
            config_hash=_config_hash(
                {
                    "kind": "html",
                    "name": name,
                    "default_model": default_model,
                }
            ),
        )
        super().__init__(backend=backend)


__all__ = ["HTMLLLMClient"]
