"""# Clients / Llama CPP Server Client.

## Introduction
Local serving with llama.cpp is a practical path for controllable offline experimentation, OpenAI-style
response contracts improve interchangeability, and HELM motivates standardized evaluation conditions. This
example validates the llama.cpp server client path with tracing and deterministic output framing.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``LlamaCppServerLLMClient.generate(...)`` with a
   fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["LlamaCppServerLLMClient.generate(...)"]
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
       "api_model": "qwen2.5-1.5b-q4",
       "base_url": null,
       "default_model": "qwen2.5-1.5b-q4",
       "host": "127.0.0.1",
       "kind": "llama_cpp_server",
       "max_retries": 3,
       "model_patterns": [
         "qwen2.5-*",
         "qwen2-*"
       ],
       "name": "llama-local-dev",
       "port": 8011
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "LlamaCppServerLLMClient",
     "default_model": "qwen2.5-1.5b-q4",
     "example": "clients/llama_cpp_server_client.py",
     "llm_call": {
       "prompt": "In one sentence, explain a key tradeoff in engineering design reviews.",
       "response_has_text": true,
       "response_model": "qwen2.5-1.5b-q4",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Tradeoff: strict review gates improve reliability but can slow delivery speed."
     },
     "server": {
       "host": "127.0.0.1",
       "kind": "llama_cpp_server",
       "managed": true,
       "port": 8011
     },
     "trace": {
       "request_id": "example-clients-llama-cpp-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-llama-cpp-call-001.jsonl"
     }
   }


## References
- `llama.cpp llama-server docs <https://github.com/ggml-org/llama.cpp#llama-server>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from design_research_agents import Tracer
from design_research_agents.llm import LLMMessage, LLMRequest
from design_research_agents.llm.clients import LlamaCppServerLLMClient


def _build_payload() -> dict[str, object]:
    client = LlamaCppServerLLMClient(
        name="llama-local-dev",
        model="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        hf_model_repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        api_model="qwen2.5-1.5b-q4",
        host="127.0.0.1",
        port=8011,
        context_window=8192,
        startup_timeout_seconds=90.0,
        poll_interval_seconds=0.5,
        python_executable=sys.executable,
        extra_server_args=("--threads", "4", "--flash_attn", "1"),
        max_retries=3,
        model_patterns=("qwen2.5-*", "qwen2-*"),
    )
    try:
        description = client.describe()
        prompt = "In one sentence, explain a key tradeoff in engineering design reviews."
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
    """Run traced llama-cpp client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-llama-cpp-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesLlamaCppClientCall",
        request_id=request_id,
        input_payload={"scenario": "llama-cpp-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/llama_cpp_server_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
