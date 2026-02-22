"""Run a traced representative ``MlxLocalLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_client_call import run_representative_chat

from design_research_agents import MlxLocalLLMClient, Tracer


def _build_payload() -> dict[str, object]:
    client = MlxLocalLLMClient(
        name="mlx-local-dev",
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        default_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        quantization="4bit",
        max_retries=2,
        model_patterns=("mlx-community/*", "qwen2.5-*"),
    )
    description = client.describe()
    llm_call = run_representative_chat(
        client=client,
        prompt="Give one concise guideline for maintainable design telemetry schemas.",
        deterministic_response=("Keep schema fields stable, documented, and versioned for comparability."),
    )
    return {
        "client_class": description["client_class"],
        "default_model": description["default_model"],
        "llm_call": llm_call,
        "backend": description["backend"],
        "capabilities": description["capabilities"],
        "server": description["server"],
    }


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
