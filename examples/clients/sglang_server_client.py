"""Run a traced representative ``SglangServerLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import sys

from design_research_agents import SglangServerLLMClient
from design_research_agents.shared.example_support import (
    print_json,
    run_representative_chat,
    run_traced_callable,
    trace_info,
)


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
        backend = client._backend
        server = client._sglang_server
        llm_call = run_representative_chat(
            client=client,
            prompt="Provide one sentence on when SGLang-style serving helps local benchmarking.",
            deterministic_response=(
                "SGLang-style serving helps when you need stable local throughput for "
                "repeated tests."
            ),
        )
        return {
            "client_class": client.__class__.__name__,
            "default_model": client.default_model(),
            "llm_call": llm_call,
            "backend": {
                "name": backend.name,
                "kind": backend.kind,
                "base_url": backend.base_url,
                "max_retries": backend.max_retries,
                "model_patterns": list(backend.model_patterns),
                "request_timeout_seconds": backend._request_timeout_seconds,
            },
            "server": {
                "manage_server": server is not None,
                "host": server.host if server is not None else None,
                "port": server.port if server is not None else None,
                "base_url": server.base_url if server is not None else None,
                "model": server.model if server is not None else None,
            },
        }
    finally:
        client.close()


def main() -> None:
    """Run traced SGLang client call payload."""
    request_id = "example-clients-sglang-server-call-001"
    payload = run_traced_callable(
        agent_name="ExamplesSglangClientCall",
        request_id=request_id,
        input_payload={"scenario": "sglang-server-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/sglang_server_client.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
