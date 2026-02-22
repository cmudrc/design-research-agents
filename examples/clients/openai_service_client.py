"""Example script.

Motivation
Run a traced representative ``OpenAIServiceLLMClient`` chat call.

Diagram
```mermaid
flowchart LR
    A["Client config"] --> B["LLMRequest"]
    B --> C["openai service client response"]
    C --> D["Describe and trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `clients` use-cases and run `openai_service_client`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/clients/openai_service_client.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LLMResponse, OpenAIServiceLLMClient, Tracer
from design_research_agents.llm import LLMMessage, LLMRequest


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
    try:
        description = client.describe()
        prompt = "In one sentence, when should engineering teams use multi-agent design critique?"
        response = client.generate(
            LLMRequest(
                messages=(
                    LLMMessage(role="system", content="You are a concise engineering design assistant."),
                    LLMMessage(role="user", content=prompt),
                ),
                model=client.default_model(),
                temperature=0.0,
                max_tokens=120,
            )
        )
        llm_call = {
            "prompt": prompt,
            "response_text": response.text,
            "response_model": response.model,
            "response_provider": response.provider,
            "response_has_text": bool(response.text.strip()),
        }
        response_contract = LLMResponse(
            text=response.text,
            model=response.model,
            provider=response.provider,
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
    finally:
        client.close()


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
