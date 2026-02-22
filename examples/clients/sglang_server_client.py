"""Run a traced representative ``SglangServerLLMClient`` chat call.

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

from design_research_agents import SglangServerLLMClient, Tracer


def _build_payload() -> dict[str, object]:
    """Build one traced payload for SGLang client configuration and call metadata.

    Returns:
        JSON-serializable payload with client config, call metadata, and trace hints.
    """
    client = SglangServerLLMClient(
        name="sglang-local-dev",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        host="127.0.0.1",
        port=30000,
        manage_server=True,
        startup_timeout_seconds=90.0,
        poll_interval_seconds=0.5,
        python_executable=sys.executable,
        extra_server_args=("--tp-size", "1"),
        request_timeout_seconds=60.0,
        max_retries=3,
        model_patterns=("Qwen/*", "qwen2.5-*"),
    )
    try:
        description = client.describe()
        llm_call = run_representative_chat(
            client=client,
            prompt="Provide one sentence on when SGLang-style serving helps local benchmarking.",
            deterministic_response=(
                "SGLang-style serving helps when you need stable local throughput for repeated tests."
            ),
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
    """Run traced SGLang client call payload."""
    request_id = "example-clients-sglang-server-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesSglangClientCall",
        request_id=request_id,
        input_payload={"scenario": "sglang-server-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/sglang_server_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
