"""Run a traced representative ``OpenAIServiceLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_client_call import run_representative_chat

from design_research_agents import LLMResponse, OpenAIServiceLLMClient, Tracer


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
    description = client.describe()
    llm_call = run_representative_chat(
        client=client,
        prompt="In one sentence, when should engineering teams use multi-agent design critique?",
        deterministic_response=(
            "Use multi-agent critique when decisions have high risk and need diverse failure analysis."
        ),
    )
    response_contract = LLMResponse(
        text=str(llm_call.get("response_text", "")),
        model=str(description["default_model"]),
        provider=str(llm_call.get("response_provider") or "deterministic"),
    )
    return {
        "client_class": description["client_class"],
        "default_model": description["default_model"],
        "llm_call": llm_call,
        "llm_response_contract_preview": {
            "model": response_contract.model,
            "provider": response_contract.provider,
        },
        "backend": description["backend"],
        "capabilities": description["capabilities"],
        "server": description["server"],
    }


def main() -> None:
    """Run traced OpenAI service client call payload."""
    request_id = "example-clients-openai-service-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesOpenAIServiceClientCall",
        request_id=request_id,
        input_payload={"scenario": "openai-service-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/openai_service_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
