"""Example script.

Motivation
Run a traced representative ``MlxLocalLLMClient`` chat call.

Diagram
```mermaid
flowchart LR
    A["Client config"] --> B["LLMRequest"]
    B --> C["mlx local client response"]
    C --> D["Describe and trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `clients` use-cases and run `mlx_local_client`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/clients/mlx_local_client.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import MlxLocalLLMClient, Tracer
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    client = MlxLocalLLMClient(
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
    finally:
        client.close()


def main() -> None:
    """Run traced MLX client call payload."""
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
