"""Example script.

Motivation
Run a traced representative ``VllmServerLLMClient`` chat call.

Diagram
```mermaid
flowchart LR
    A["Client config"] --> B["LLMRequest"]
    B --> C["vllm server client response"]
    C --> D["Describe and trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `clients` use-cases and run `vllm_server_client`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/clients/vllm_server_client.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from design_research_agents import Tracer, VllmServerLLMClient
from design_research_agents.llm import LLMMessage, LLMRequest


def _build_payload() -> dict[str, object]:
    client = VllmServerLLMClient(
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
    finally:
        client.close()


def main() -> None:
    """Run traced vLLM client call payload."""
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
