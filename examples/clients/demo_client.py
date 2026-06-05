"""# Clients / Demo LLM Client.

## Introduction
``DemoLLMClient`` is the workshop-friendly local model path: it wraps a managed
llama.cpp server with Qwen3-0.6B GGUF defaults, bounded generation settings, and
non-thinking prompt controls.

## Technical Implementation
1. Construct ``DemoLLMClient`` through public APIs so the managed llama.cpp server
   lifecycle remains hidden from workshop scripts.
2. Send one ``LLMRequest`` with a short design-research prompt and bounded output.
3. Print the client description, server snapshot, and normalized response payload
   for repeatable docs and smoke-test output.

## Expected Results
The example prints one JSON payload with client configuration and a single
response. Under deterministic example tests, the model call is monkeypatched.

## References
- `Qwen3-0.6B model card <https://huggingface.co/Qwen/Qwen3-0.6B>`_
- `Qwen3-0.6B GGUF model card <https://huggingface.co/Qwen/Qwen3-0.6B-GGUF>`_
- `Qwen llama.cpp local run guide <https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag


def _build_payload() -> dict[str, object]:
    """Run one direct DemoLLMClient call.

    Returns:
        JSON-serializable example payload.
    """
    with drag.DemoLLMClient(name="demo-qwen3-workshop") as client:
        description = client.describe()
        prompt = "In one sentence, name a useful design-research workshop activity."
        response = client.generate(
            drag.LLMRequest(
                messages=(
                    drag.LLMMessage(role="system", content="You are a concise design research assistant."),
                    drag.LLMMessage(role="user", content=prompt),
                ),
                model=client.default_model(),
                max_tokens=80,
            )
        )
        return {
            "client_class": description["client_class"],
            "default_model": description["default_model"],
            "backend": description["backend"],
            "capabilities": description["capabilities"],
            "server": description["server"],
            "llm_call": {
                "prompt": prompt,
                "response_text": response.text,
                "response_model": response.model,
                "response_provider": response.provider,
                "response_has_text": bool(response.text.strip()),
            },
        }


def main() -> None:
    """Run the traced DemoLLMClient example."""
    request_id = "example-clients-demo-call-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesDemoClientCall",
        request_id=request_id,
        input_payload={"scenario": "demo-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/demo_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
