"""Run a traced representative ``MlxLocalLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import MlxLocalLLMClient
from design_research_agents.shared.example_support import (
    print_json,
    run_representative_chat,
    run_traced_callable,
    trace_info,
)


def _build_payload() -> dict[str, object]:
    client = MlxLocalLLMClient(
        name="mlx-local-dev",
        model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        default_model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        quantization="4bit",
        max_retries=2,
        model_patterns=("mlx-community/*", "qwen2.5-*"),
    )
    backend = client._backend
    capabilities = backend.capabilities()
    llm_call = run_representative_chat(
        client=client,
        prompt="Give one concise guideline for maintainable design telemetry schemas.",
        deterministic_response=(
            "Keep schema fields stable, documented, and versioned for comparability."
        ),
    )
    return {
        "client_class": client.__class__.__name__,
        "default_model": client.default_model(),
        "llm_call": llm_call,
        "backend": {
            "name": backend.name,
            "kind": backend.kind,
            "model_id": backend._model_id,
            "quantization": backend._quantization,
            "max_retries": backend.max_retries,
            "model_patterns": list(backend.model_patterns),
        },
        "capabilities": {
            "streaming": capabilities.streaming,
            "tool_calling": capabilities.tool_calling,
            "json_mode": capabilities.json_mode,
            "vision": capabilities.vision,
            "max_context_tokens": capabilities.max_context_tokens,
        },
    }


def main() -> None:
    """Run traced MLX client call payload."""
    request_id = "example-clients-mlx-local-call-001"
    payload = run_traced_callable(
        agent_name="ExamplesMlxClientCall",
        request_id=request_id,
        input_payload={"scenario": "mlx-local-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/mlx_local_client.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
