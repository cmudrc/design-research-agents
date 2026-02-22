"""Run a traced representative ``TransformersLocalLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import TransformersLocalLLMClient
from design_research_agents._shared._example_support import (
    print_json,
    run_representative_chat,
    run_traced_callable,
    trace_info,
)


def _build_payload() -> dict[str, object]:
    client = TransformersLocalLLMClient(
        name="transformers-local-dev",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        default_model="Qwen/Qwen2.5-1.5B-Instruct",
        device="auto",
        dtype="auto",
        quantization="none",
        trust_remote_code=False,
        revision="main",
        max_retries=2,
        model_patterns=("Qwen/*", "qwen2.5-*"),
    )
    backend = client._backend
    capabilities = backend.capabilities()
    llm_call = run_representative_chat(
        client=client,
        prompt="Provide one sentence on why deterministic local runs aid design reproducibility.",
        deterministic_response=("Deterministic local runs make design comparisons repeatable across experiments."),
    )
    return {
        "client_class": client.__class__.__name__,
        "default_model": client.default_model(),
        "llm_call": llm_call,
        "backend": {
            "name": backend.name,
            "kind": backend.kind,
            "model_id": backend._model_id,
            "device": backend._device,
            "dtype": backend._dtype,
            "quantization": backend._quantization,
            "trust_remote_code": backend._trust_remote_code,
            "revision": backend._revision,
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
    """Run traced Transformers client call payload."""
    request_id = "example-clients-transformers-local-call-001"
    payload = run_traced_callable(
        agent_name="ExamplesTransformersClientCall",
        request_id=request_id,
        input_payload={"scenario": "transformers-local-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/transformers_local_client.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
