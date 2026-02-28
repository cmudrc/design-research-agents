"""# Clients / MLX Local Client.

## Introduction
MLX-LM provides an Apple-silicon-native local inference stack, HELM motivates reproducible evaluation
baselines, and AI-assisted design synthesis work connects these runtimes to educational design workflows.
This example exercises the MLX local client path with trace artifacts suitable for repeatable comparisons.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MLXLocalLLMClient.generate(...)`` with a fixed
   ``request_id``.
3. Construct ``LLMRequest`` inputs and call ``generate`` through the selected client implementation.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["MLXLocalLLMClient.generate(...)"]
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
       "base_url": null,
       "default_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
       "kind": "mlx_local",
       "max_retries": 2,
       "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
       "model_patterns": [
         "mlx-community/*",
         "qwen2.5-*"
       ],
       "name": "mlx-local-dev",
       "quantization": "4bit"
     },
     "capabilities": {
       "json_mode": "prompt+validate",
       "max_context_tokens": null,
       "streaming": false,
       "tool_calling": "best_effort",
       "vision": false
     },
     "client_class": "MLXLocalLLMClient",
     "default_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
     "example": "clients/mlx_local_client.py",
     "llm_call": {
       "prompt": "Give one concise guideline for maintainable design telemetry schemas.",
       "response_has_text": true,
       "response_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
       "response_provider": "example-test-monkeypatch",
       "response_text": "Keep schema fields stable, documented, and versioned for comparability."
     },
     "server": null,
     "trace": {
       "request_id": "example-clients-mlx-local-call-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162206Z_example-clients-mlx-local-call-001.jsonl"
     }
   }


## References
- `MLX-LM <https://github.com/ml-explore/mlx-lm>`_
- `Holistic Evaluation of Language Models (HELM) <https://arxiv.org/abs/2211.09110>`_
- `AI-assisted design synthesis and human creativity in engineering education <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1714523/full>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import MLXLocalLLMClient, Tracer
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    client = MLXLocalLLMClient(
        name="mlx-local-dev",
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        default_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        quantization="4bit",
        max_retries=2,
        model_patterns=("mlx-community/*", "qwen2.5-*"),
    )
    try:
        description = client.describe()
        prompt = "Give one concise guideline for maintainable design telemetry schemas."
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
    """Run traced MLX client call payload."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-clients-mlx-local-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesMlxClientCall",
        request_id=request_id,
        input_payload={"scenario": "mlx-local-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/mlx_local_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
