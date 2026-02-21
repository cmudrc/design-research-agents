"""Run a traced representative ``OllamaLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from design_research_agents import OllamaLLMClient
from design_research_agents.shared.example_support import (
    print_json,
    run_representative_chat,
    run_traced_callable,
    trace_info,
)


def _build_payload() -> dict[str, object]:
    """Build one traced payload for Ollama client configuration and call metadata.

    Returns:
        JSON-serializable payload with client config, call metadata, and trace hints.
    """
    client = OllamaLLMClient(
        name="ollama-local-dev",
        default_model="qwen2.5:1.5b-instruct",
        host="127.0.0.1",
        port=11434,
        manage_server=True,
        ollama_executable="ollama",
        auto_pull_model=False,
        startup_timeout_seconds=60.0,
        poll_interval_seconds=0.25,
        request_timeout_seconds=60.0,
        max_retries=2,
        model_patterns=("qwen2.5:*", "llama3:*"),
    )
    try:
        backend = client._backend
        server = client._ollama_server
        llm_call = run_representative_chat(
            client=client,
            prompt="Give one sentence on when to use local model pull automation.",
            deterministic_response=(
                "Use automated local pulls when startup reliability matters more than "
                "cold-start time."
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
                "auto_pull_model": server.auto_pull_model if server is not None else None,
                "default_model": server.default_model if server is not None else None,
            },
        }
    finally:
        client.close()


def main() -> None:
    """Run traced Ollama client call payload."""
    request_id = "example-clients-ollama-local-call-001"
    payload = run_traced_callable(
        agent_name="ExamplesOllamaClientCall",
        request_id=request_id,
        input_payload={"scenario": "ollama-local-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/ollama_local_client.py"
    payload["trace"] = trace_info(request_id)
    print_json(payload)


if __name__ == "__main__":
    main()
