"""Run a traced representative ``LlamaCppServerLLMClient`` chat call.

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

from design_research_agents import Tracer
from design_research_agents.llm.clients import LlamaCppServerLLMClient


def _build_payload() -> dict[str, object]:
    client = LlamaCppServerLLMClient(
        name="llama-local-dev",
        model="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        hf_model_repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        api_model="qwen2.5-1.5b-q4",
        host="127.0.0.1",
        port=8011,
        context_window=8192,
        startup_timeout_seconds=90.0,
        poll_interval_seconds=0.5,
        python_executable=sys.executable,
        extra_server_args=("--threads", "4", "--flash_attn", "1"),
        max_retries=3,
        model_patterns=("qwen2.5-*", "qwen2-*"),
    )
    try:
        description = client.describe()
        llm_call = run_representative_chat(
            client=client,
            prompt="In one sentence, explain a key tradeoff in engineering design reviews.",
            deterministic_response=("Tradeoff: strict review gates improve reliability but can slow delivery speed."),
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
    """Run traced llama-cpp client call payload."""
    request_id = "example-clients-llama-cpp-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesLlamaCppClientCall",
        request_id=request_id,
        input_payload={"scenario": "llama-cpp-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/llama_cpp_server_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
