"""Configure ``TransformersLocalLLMClient`` with explicit local inference options."""

from __future__ import annotations

import json

from design_research_agents import TransformersLocalLLMClient


def main() -> None:
    """Build a fully configured Transformers local client and print settings."""
    client = TransformersLocalLLMClient(
        name="transformers-local-dev",
        model_id="distilgpt2",
        default_model="distilgpt2",
        device="cpu",
        dtype="float32",
        quantization="none",
        trust_remote_code=False,
        revision="main",
        max_retries=2,
        model_patterns=("distilgpt2", "gpt2*"),
    )
    backend = client._backend
    capabilities = backend.capabilities()
    payload = {
        "client_class": client.__class__.__name__,
        "default_model": client.default_model(),
        "backend": {
            "name": backend.name,
            "kind": backend.kind,
            "model_id": backend._model_id,
            "device": backend._device,
            "dtype": backend._dtype,
            "quantization": backend._quantization,
            "trust_remote_code": backend._trust_remote_code,
            "revision": backend._revision,
            "max_retries": backend.max_retries,
            "model_patterns": list(backend.model_patterns),
        },
        "capabilities": {
            "streaming": capabilities.streaming,
            "tool_calling": capabilities.tool_calling,
            "json_mode": capabilities.json_mode,
            "vision": capabilities.vision,
            "max_context_tokens": capabilities.max_context_tokens,
        },
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
