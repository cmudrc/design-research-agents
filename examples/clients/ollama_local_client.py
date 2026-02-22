"""Run a traced representative ``OllamaLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_client_call import run_representative_chat

from design_research_agents import OllamaLLMClient, Tracer


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
        description = client.describe()
        llm_call = run_representative_chat(
            client=client,
            prompt="Give one sentence on when to use local model pull automation.",
            deterministic_response=(
                "Use automated local pulls when startup reliability matters more than cold-start time."
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
    """Run traced Ollama client call payload."""
    request_id = "example-clients-ollama-local-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesOllamaClientCall",
        request_id=request_id,
        input_payload={"scenario": "ollama-local-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/ollama_local_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
