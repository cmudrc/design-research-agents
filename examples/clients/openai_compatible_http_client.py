"""Configure ``OpenAICompatibleHTTPLLMClient`` for local or remote endpoints."""

from __future__ import annotations

import json

from design_research_agents import OpenAICompatibleHTTPLLMClient


def main() -> None:
    """Build a fully configured OpenAI-compatible HTTP client and print settings."""
    client = OpenAICompatibleHTTPLLMClient(
        name="local-openai-compat",
        base_url="http://127.0.0.1:8011/v1",
        default_model="qwen2.5-1.5b-q4",
        api_key_env="OPENAI_API_KEY",
        api_key="example-key-for-config-demo",
        max_retries=3,
        model_patterns=("qwen2.5-*", "qwen2-*"),
    )
    backend = client._backend
    capabilities = backend.capabilities()
    payload = {
        "client_class": client.__class__.__name__,
        "default_model": client.default_model(),
        "backend": {
            "name": backend.name,
            "kind": backend.kind,
            "base_url": backend.base_url,
            "chat_url": backend._chat_url,
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
