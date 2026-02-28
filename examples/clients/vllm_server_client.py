"""# Clients / vLLM Server Client.

## Introduction
vLLM is a common high-performance inference server, OpenAI-compatible response contracts enable drop-in
orchestration reuse, and HELM provides context for why consistent serving interfaces help evaluation. This
example exercises the vLLM server client integration with explicit trace reporting.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``VLLMServerLLMClient.generate(...)`` with a
   fixed ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["VLLMServerLLMClient.generate(...)"]
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
       "base_url": "http://127.0.0.1:8002/v1",
       "default_model": "qwen2.5-1.5b-instruct",
       "host": "127.0.0.1",
       "kind": "vllm_server",
       "max_retries": 3,
       "model_patterns": [
         "qwen2.5-*"
       ],
       "name": "vllm-local-dev",
       "port": 8002
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "VLLMServerLLMClient",
     "default_model": "qwen2.5-1.5b-instruct",
     "example": "clients/vllm_server_client.py",
     "llm_call": {
       "prompt": "Provide one sentence on why local serving helps reproducible benchmarking.",
       "response_has_text": true,
       "response_model": "qwen2.5-1.5b-instruct",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Local serving reduces backend drift and improves benchmark reproducibility."
     },
     "server": {
       "host": "127.0.0.1",
       "kind": "vllm_server",
       "managed": true,
       "port": 8002
     },
     "trace": {
       "request_id": "example-clients-vllm-server-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-vllm-server-call-001.jsonl"
     }
   }


## References
- `vLLM OpenAI-Compatible Server <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from design_research_agents import Tracer, VLLMServerLLMClient
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    client = VLLMServerLLMClient(
        name="vllm-local-dev",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_model="qwen2.5-1.5b-instruct",
        host="127.0.0.1",
        port=8002,
        manage_server=True,
        startup_timeout_seconds=90.0,
        poll_interval_seconds=0.5,
        python_executable=sys.executable,
        extra_server_args=("--dtype", "auto"),
        request_timeout_seconds=60.0,
        max_retries=3,
        model_patterns=("qwen2.5-*",),
    )
    try:
        description = client.describe()
        prompt = "Provide one sentence on why local serving helps reproducible benchmarking."
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
    """Run traced vLLM client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-vllm-server-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesVllmClientCall",
        request_id=request_id,
        input_payload={"scenario": "vllm-server-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/vllm_server_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
