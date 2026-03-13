"""# Clients / Groq Service Client.

## Introduction
Groq hosted inference can provide low-latency responses for agent loops that still need standard chat-completion
semantics such as streaming and tool-call metadata. This example runs the Groq service client through the same
framework contracts used by other providers, with trace artifacts suitable for regression checks.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output for repeatable diagnostics.
2. Build one request using public APIs and execute ``GroqServiceLLMClient.generate(...)``.
3. Serialize the key response contract fields and backend metadata into one JSON payload.
4. Emit the payload with fixed request id metadata for deterministic documentation tests.

```mermaid
flowchart LR
    A["Prompt input"] --> B["main(): tracing setup"]
    B --> C["GroqServiceLLMClient.generate(...)"]
    C --> D["LLMRequest and LLMResponse contracts"]
    C --> E["Tracer lifecycle events"]
    D --> F["Output payload"]
    E --> F
    F --> G["Printed JSON result"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "backend": {
       "api_key_env": "GROQ_API_KEY",
       "base_url": "https://api.groq.com",
       "default_model": "llama-3.1-8b-instant",
       "kind": "groq_service",
       "max_retries": 3,
       "model_patterns": [
         "llama-3.1-8b-instant",
         "llama-3.1-*"
       ],
       "name": "groq-prod"
     },
     "capabilities": {
       "json_mode": "native",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "native",
       "vision": false
     },
     "client_class": "GroqServiceLLMClient",
     "default_model": "llama-3.1-8b-instant",
     "example": "clients/groq_service_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on when teams should trade latency for review depth.",
       "response_has_text": true,
       "response_model": "llama-3.1-8b-instant",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Prefer deeper review when architectural choices are expensive to reverse."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-groq-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-groq-service-call-001.jsonl"
     }
   }


## References
- `Groq Python SDK repository <https://github.com/groq/groq-python>`_
- `Groq model catalog docs <https://console.groq.com/docs/models>`_
- `Groq chat completion docs <https://console.groq.com/docs/text-chat>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def _build_payload() -> dict[str, object]:
    # Build the hosted Groq client using public runtime APIs, then execute one representative request.
    client = drag.GroqServiceLLMClient(
        name="groq-prod",
        default_model="llama-3.1-8b-instant",
        api_key_env="GROQ_API_KEY",
        api_key="example-key-for-config-demo",
        base_url="https://api.groq.com",
        max_retries=3,
        model_patterns=("llama-3.1-8b-instant", "llama-3.1-*"),
    )
    description = client.describe()
    prompt = "Provide one sentence on when teams should trade latency for review depth."
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
    return {
        "client_class": description["client_class"],
        "default_model": description["default_model"],
        "llm_call": llm_call,
        "backend": description["backend"],
        "capabilities": description["capabilities"],
        "server": description["server"],
    }


def main() -> None:
    """Run traced Groq service client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-groq-service-call-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesGroqServiceClientCall",
        request_id=request_id,
        input_payload={"scenario": "groq-service-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/groq_service_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    # Print the results
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
