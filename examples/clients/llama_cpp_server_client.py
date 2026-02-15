"""Configure ``LlamaCppServerLLMClient`` with explicit local-server settings."""

from __future__ import annotations

import json
import sys

from design_research_agents.llm.clients import LlamaCppServerLLMClient


def main() -> None:
    """Build a fully configured local llama-cpp client and print settings."""
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
        backend = client._backend
        server = client._llama_server
        payload = {
            "client_class": client.__class__.__name__,
            "default_model": client.default_model(),
            "backend": {
                "name": backend.name,
                "kind": backend.kind,
                "max_retries": backend.max_retries,
                "model_patterns": list(backend.model_patterns),
            },
            "server": {
                "python_executable": server.python_executable,
                "host": server.host,
                "port": server.port,
                "base_url": server.base_url,
                "model": server.model,
                "hf_model_repo_id": server.hf_model_repo_id,
                "api_model": server.api_model,
                "startup_timeout_seconds": server.startup_timeout_seconds,
                "poll_interval_seconds": server.poll_interval_seconds,
                "extra_server_args": list(server.extra_server_args),
            },
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    finally:
        client.close()


if __name__ == "__main__":
    main()
