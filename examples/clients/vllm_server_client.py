"""Run a traced representative ``VllmServerLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _support_client_call import run_representative_chat

from design_research_agents import Tracer, VllmServerLLMClient


def _build_payload() -> dict[str, object]:
    """Build one traced payload for vLLM client configuration and call metadata.

    Returns:
        JSON-serializable payload with client config, call metadata, and trace hints.
    """
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
        llm_call = run_representative_chat(
            client=client,
            prompt="Provide one sentence on why local serving helps reproducible benchmarking.",
            deterministic_response=("Local serving reduces backend drift and improves benchmark reproducibility."),
        )
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
