"""Run a traced representative ``TransformersLocalLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_client_call import run_representative_chat

from design_research_agents import Tracer, TransformersLocalLLMClient


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
    description = client.describe()
    llm_call = run_representative_chat(
        client=client,
        prompt="Provide one sentence on why deterministic local runs aid design reproducibility.",
        deterministic_response=("Deterministic local runs make design comparisons repeatable across experiments."),
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
    """Run traced Transformers client call payload."""
    request_id = "example-clients-transformers-local-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesTransformersClientCall",
        request_id=request_id,
        input_payload={"scenario": "transformers-local-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/transformers_local_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
