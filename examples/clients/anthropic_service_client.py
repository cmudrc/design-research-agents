"""# Clients / Anthropic Service Client.

## Introduction
Anthropic hosted inference is useful when teams want strong instruction-following and tool-use support from one
managed API while keeping application code on provider-neutral LLM contracts. This example exercises the
Anthropic service client path with trace capture and deterministic output support for CI.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console sinks so each run emits machine-readable traces.
2. Build runtime inputs through public package APIs and invoke ``AnthropicServiceLLMClient.generate(...)``.
3. Construct ``LLMRequest`` payload fields and execute one representative remote-style call.
4. Print a compact JSON payload that includes trace metadata for docs and deterministic tests.

```mermaid
flowchart LR
    A["Prompt input"] --> B["main(): tracing setup"]
    B --> C["AnthropicServiceLLMClient.generate(...)"]
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
       "api_key_env": "ANTHROPIC_API_KEY",
       "base_url": "https://api.anthropic.com",
       "default_model": "claude-3-5-haiku-latest",
       "kind": "anthropic_service",
       "max_retries": 3,
       "model_patterns": [
         "claude-3-5-haiku-latest",
         "claude-3-5-*"
       ],
       "name": "anthropic-prod"
     },
     "capabilities": {
       "json_mode": "native",
       "max_context_tokens": null,
       "streaming": true,
       "tool_calling": "native",
       "vision": false
     },
     "client_class": "AnthropicServiceLLMClient",
     "default_model": "claude-3-5-haiku-latest",
     "example": "clients/anthropic_service_client.py",
     "llm_call": {
       "prompt": "In one sentence, when should teams run architecture red-team reviews?",
       "response_has_text": true,
       "response_model": "claude-3-5-haiku-latest",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Run architecture red-team reviews before committing high-impact changes with uncertain failure modes."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-anthropic-service-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-anthropic-service-call-001.jsonl"
     }
   }


## References
- `Anthropic API docs <https://platform.claude.com/docs/en/api/overview>`_
- `Anthropic Python SDK repository <https://github.com/anthropics/anthropic-sdk-python>`_
- `Anthropic tool use docs <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import AnthropicServiceLLMClient, Tracer
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    client = AnthropicServiceLLMClient(
        name="anthropic-prod",
        default_model="claude-3-5-haiku-latest",
        api_key_env="ANTHROPIC_API_KEY",
        api_key="example-key-for-config-demo",
        base_url="https://api.anthropic.com",
        max_retries=3,
        model_patterns=("claude-3-5-haiku-latest", "claude-3-5-*"),
    )
    description = client.describe()
    prompt = "In one sentence, when should teams run architecture red-team reviews?"
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
    return {
        "client_class": description["client_class"],
        "default_model": description["default_model"],
        "llm_call": llm_call,
        "backend": description["backend"],
        "capabilities": description["capabilities"],
        "server": description["server"],
    }


def main() -> None:
    """Run traced Anthropic service client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-anthropic-service-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesAnthropicServiceClientCall",
        request_id=request_id,
        input_payload={"scenario": "anthropic-service-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/anthropic_service_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
