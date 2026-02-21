"""Run a traced representative ``OpenAICompatibleHTTPLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import OpenAICompatibleHTTPLLMClient
from design_research_agents.shared.example_support import (
    print_json,
    run_representative_chat,
    run_traced_callable,
    trace_info,
)


def _build_payload() -> dict[str, object]:
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
    llm_call = run_representative_chat(
        client=client,
        prompt="Provide one sentence on balancing latency and quality in design review assistants.",
        deterministic_response=(
            "Use fast drafts for iteration, then escalate critical decisions to "
            "higher-quality models."
        ),
    )
    return {
        "client_class": client.__class__.__name__,
        "default_model": client.default_model(),
        "llm_call": llm_call,
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


def main() -> None:
    """Run traced OpenAI-compatible client call payload."""
    request_id = "example-clients-openai-compatible-call-001"
    payload = run_traced_callable(
        agent_name="ExamplesOpenAICompatClientCall",
        request_id=request_id,
        input_payload={"scenario": "openai-compatible-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/openai_compatible_http_client.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
