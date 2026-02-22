r"""# Clients / Ollama Local Client.

## Introduction
Ollama operationalizes local model serving, the OpenAI Responses API provides a common contract surface, and
HELM underlines why comparable execution conditions matter in benchmarking. This example verifies the Ollama
client integration path under the project tracing/runtime conventions.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``OllamaLLMClient.generate(...)`` with a fixed
   ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["OllamaLLMClient.generate(...)"]
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
       "base_url": "http://127.0.0.1:11434",
       "default_model": "qwen2.5:1.5b-instruct",
       "host": "127.0.0.1",
       "kind": "ollama",
       "max_retries": 2,
       "model_patterns": [
         "qwen2.5:*",
         "llama3:*"
       ],
       "name": "ollama-local-dev",
       "port": 11434
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "OllamaLLMClient",
     "default_model": "qwen2.5:1.5b-instruct",
     "example": "clients/ollama_local_client.py",
     "llm_call": {
       "prompt": "Give one sentence on when to use local model pull automation.",
       "response_has_text": true,
       "response_model": "qwen2.5:1.5b-instruct",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Use automated local pulls when startup reliability matters more than cold-start time."
     },
     "server": {
       "host": "127.0.0.1",
       "kind": "ollama",
       "managed": true,
       "port": 11434
     },
     "trace": {
       "request_id": "example-clients-ollama-local-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-ollama-local-call-001.jsonl"
     }
   }


## References
- `Ollama API Docs <https://docs.ollama.com/api>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import OllamaLLMClient, Tracer
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    client = OllamaLLMClient(
        name="ollama-local-dev",
        default_model="qwen2.5:1.5b-instruct",
        host="127.0.0.1",
        port=11434,
        manage_server=True,
        ollama_executable="ollama",
        auto_pull_model=False,
        startup_timeout_seconds=60.0,
        poll_interval_seconds=0.25,
        request_timeout_seconds=60.0,
        max_retries=2,
        model_patterns=("qwen2.5:*", "llama3:*"),
    )
    try:
        description = client.describe()
        prompt = "Give one sentence on when to use local model pull automation."
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
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        client.close()


def main() -> None:
    """Run traced Ollama client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-ollama-local-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesOllamaClientCall",
        request_id=request_id,
        input_payload={"scenario": "ollama-local-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/ollama_local_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
