"""Configure ``OpenAIServiceLLMClient`` with explicit service settings."""

from __future__ import annotations

import json

from design_research_agents import OpenAIServiceLLMClient


def main() -> None:
    """Build a fully configured OpenAI service client and print settings."""
    client = OpenAIServiceLLMClient(
        name="openai-prod",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        api_key="example-key-for-config-demo",
        base_url="https://api.openai.com/v1",
        max_retries=4,
        model_patterns=("gpt-4o-mini", "gpt-4o-*"),
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
