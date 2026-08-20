"""# Clients / Gemini Service Client.

## Introduction
Gemini hosted inference is useful when teams want multimodel experimentation through one provider SDK, while
keeping request payloads under the framework's provider-neutral LLM contracts. This example exercises the
Gemini service client path with trace capture and deterministic output support for CI.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console sinks so each run emits machine-readable traces.
2. Build runtime inputs through public package APIs and invoke ``GeminiServiceLLMClient.generate(...)``.
3. Construct ``LLMRequest`` payload fields and execute one representative remote-style call.
4. Print a compact JSON payload that includes trace metadata for docs and deterministic tests.

```mermaid
flowchart LR
    A["Prompt input"] --> B["main(): tracing setup"]
    B --> C["GeminiServiceLLMClient.generate(...)"]
    C --> D["LLMRequest and LLMResponse contracts"]
    C --> E["Tracer JSONL + console events"]
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
       "api_key_env": "GOOGLE_API_KEY",
       "default_model": "gemini-2.5-flash",
       "kind": "gemini_service",
       "max_retries": 3,
       "model_patterns": [
         "gemini-2.5-flash",
         "gemini-2.5-*"
       ],
       "name": "gemini-prod"
     },
     "capabilities": {
       "json_mode": "native",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "none",
       "vision": false
     },
     "client_class": "GeminiServiceLLMClient",
     "default_model": "gemini-2.5-flash",
     "example": "clients/gemini_service_client.py",
     "llm_call": {
       "prompt": "In one sentence, when should engineers run an explicit design pre-mortem?",
       "response_has_text": true,
       "response_model": "gemini-2.5-flash",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Run a design pre-mortem before committing architecture changes with high uncertainty or safety risk."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-gemini-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-gemini-service-call-001.jsonl"
     }
   }


## References
- `Google Gen AI Python SDK docs <https://googleapis.github.io/python-genai/>`_
- `Gemini API key setup guide <https://ai.google.dev/gemini-api/docs/api-key>`_
- `Gemini API model docs <https://ai.google.dev/gemini-api/docs/models>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def _build_payload() -> dict[str, object]:
    # Build the hosted Gemini client using public runtime APIs, then execute one representative request.
    client = drag.GeminiServiceLLMClient(
        name="gemini-prod",
        default_model="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
        max_retries=3,
        model_patterns=("gemini-2.5-flash", "gemini-2.5-*"),
    )
    description = client.describe()
    prompt = "In one sentence, when should engineers run an explicit design pre-mortem?"
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
    """Run traced Gemini service client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-gemini-service-call-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesGeminiServiceClientCall",
        request_id=request_id,
        input_payload={"scenario": "gemini-service-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/gemini_service_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    # Print the results
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
