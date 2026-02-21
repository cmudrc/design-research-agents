"""Run a traced representative ``OpenAIServiceLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import OpenAIServiceLLMClient
from design_research_agents.shared.example_support import (
    print_json,
    run_representative_chat,
    run_traced_callable,
    trace_info,
)


def _build_payload() -> dict[str, object]:
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
    llm_call = run_representative_chat(
        client=client,
        prompt="In one sentence, when should engineering teams use multi-agent design critique?",
        deterministic_response=(
            "Use multi-agent critique when decisions have high risk and need "
            "diverse failure analysis."
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
    """Run traced OpenAI service client call payload."""
    request_id = "example-clients-openai-service-call-001"
    payload = run_traced_callable(
        agent_name="ExamplesOpenAIServiceClientCall",
        request_id=request_id,
        input_payload={"scenario": "openai-service-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/openai_service_client.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
