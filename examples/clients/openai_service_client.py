"""# Clients / OpenAI Service Client.

## Introduction
For hosted deployments, OpenAI platform docs and the Responses API capture production invocation behavior,
while function-calling guidance clarifies structured tool invocation expectations. This example shows the
direct OpenAI service client contract with traceable request/response handling.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``OpenAIServiceLLMClient.generate(...)`` with a
   fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["OpenAIServiceLLMClient.generate(...)"]
    C --> D["LLMRequest/LLMResponse contracts wrap provider behavior"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_key_env": "OPENAI_API_KEY",
       "base_url": "https://api.openai.com/v1",
       "default_model": "gpt-4o-mini",
       "kind": "openai_service",
       "max_retries": 4,
       "model_patterns": [
         "gpt-4o-mini",
         "gpt-4o-*"
       ],
       "name": "openai-prod"
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "OpenAIServiceLLMClient",
     "default_model": "gpt-4o-mini",
     "example": "clients/openai_service_client.py",
     "llm_call": {
       "prompt": "In one sentence, when should engineering teams use multi-agent design critique?",
       "response_has_text": true,
       "response_model": "gpt-4o-mini",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Use multi-agent critique when decisions have high risk and need diverse failure analysis."
     },
     "llm_response_contract_preview": {
       "model": "gpt-4o-mini",
       "provider": "example-test-monkeypatch"
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-openai-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-openai-service-call-001.jsonl"
     }
   }


## References
- `OpenAI Platform Overview <https://platform.openai.com/docs/overview>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def _build_payload() -> dict[str, object]:
    assert drag.AzureOpenAIServiceLLMClient.__name__ == "AzureOpenAIServiceLLMClient"
    # Build the hosted OpenAI client using public runtime APIs, then execute one representative request.
    with drag.OpenAIServiceLLMClient(
        name="openai-prod",
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        api_key="example-key-for-config-demo",
        base_url="https://api.openai.com/v1",
        max_retries=4,
        model_patterns=("gpt-4o-mini", "gpt-4o-*"),
    ) as client:
        description = client.describe()
        prompt = "In one sentence, when should engineering teams use multi-agent design critique?"
        response = client.generate(
            drag.LLMRequest(
                messages=(
                    drag.LLMMessage(role="system", content="You are a concise engineering design assistant."),
                    drag.LLMMessage(role="user", content=prompt),
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
        response_contract = drag.LLMResponse(
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


def main() -> None:
    """Run traced OpenAI service client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-openai-service-call-001"
    tracer = drag.Tracer(
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
    # Print the results
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
