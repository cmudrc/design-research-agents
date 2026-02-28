"""# Clients / OpenAI Compatible HTTP Client.

## Introduction
OpenAI-compatible HTTP surfaces are valuable because they let one orchestration stack target multiple
providers; vLLM and SGLang both expose this style of interface while OpenAI Responses API defines the
baseline semantics. This example demonstrates that compatibility layer in the framework client runtime.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``OpenAICompatibleHTTPLLMClient.generate(...)``
   with a fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["OpenAICompatibleHTTPLLMClient.generate(...)"]
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
       "base_url": "http://127.0.0.1:8011/v1",
       "default_model": "qwen2.5-1.5b-q4",
       "kind": "openai_compatible_http",
       "max_retries": 3,
       "model_patterns": [
         "qwen2.5-*",
         "qwen2-*"
       ],
       "name": "local-openai-compat"
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "OpenAICompatibleHTTPLLMClient",
     "default_model": "qwen2.5-1.5b-q4",
     "example": "clients/openai_compatible_http_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on balancing latency and quality in design review assistants.",
       "response_has_text": true,
       "response_model": "qwen2.5-1.5b-q4",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Use fast drafts for iteration, then escalate critical decisions to higher-quality models."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-openai-compatible-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-openai-compatible-call-001.jsonl"
     }
   }


## References
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `vLLM OpenAI-Compatible Server <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>`_
- `SGLang OpenAI-Compatible API <https://docs.sglang.ai/basic_usage/openai_api.html>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import OpenAICompatibleHTTPLLMClient, Tracer
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    with OpenAICompatibleHTTPLLMClient(
        name="local-openai-compat",
        base_url="http://127.0.0.1:8011/v1",
        default_model="qwen2.5-1.5b-q4",
        api_key_env="OPENAI_API_KEY",
        api_key="example-key-for-config-demo",
        max_retries=3,
        model_patterns=("qwen2.5-*", "qwen2-*"),
    ) as client:
        description = client.describe()
        prompt = "Provide one sentence on balancing latency and quality in design review assistants."
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
    """Run traced OpenAI-compatible client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-openai-compatible-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesOpenAICompatClientCall",
        request_id=request_id,
        input_payload={"scenario": "openai-compatible-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/openai_compatible_http_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
