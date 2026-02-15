"""Configure ``MlxLocalLLMClient`` with explicit local MLX settings."""

from __future__ import annotations

import json

from design_research_agents import MlxLocalLLMClient


def main() -> None:
    """Build a fully configured MLX local client and print settings."""
    client = MlxLocalLLMClient(
        name="mlx-local-dev",
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        default_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        quantization="4bit",
        max_retries=2,
        model_patterns=("mlx-community/*", "qwen2.5-*"),
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
            "quantization": backend._quantization,
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
