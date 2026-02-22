"""Run a traced representative ``OpenAICompatibleHTTPLLMClient`` chat call.

Expected observations:
- output includes one representative chat completion under ``llm_call``.
- ``llm_call.response_has_text`` is ``true``.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_client_call import run_representative_chat

from design_research_agents import OpenAICompatibleHTTPLLMClient, Tracer


def _build_payload() -> dict[str, object]:
    client = OpenAICompatibleHTTPLLMClient(
        name="local-openai-compat",
        base_url="http://127.0.0.1:8011/v1",
        default_model="qwen2.5-1.5b-q4",
        api_key_env="OPENAI_API_KEY",
        api_key="example-key-for-config-demo",
        max_retries=3,
        model_patterns=("qwen2.5-*", "qwen2-*"),
    )
    description = client.describe()
    llm_call = run_representative_chat(
        client=client,
        prompt="Provide one sentence on balancing latency and quality in design review assistants.",
        deterministic_response=(
            "Use fast drafts for iteration, then escalate critical decisions to higher-quality models."
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


def main() -> None:
    """Run traced OpenAI-compatible client call payload."""
    request_id = "example-clients-openai-compatible-call-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    payload = tracer.run_callable(
        agent_name="ExamplesOpenAICompatClientCall",
        request_id=request_id,
        input_payload={"scenario": "openai-compatible-client-call"},
        function=_build_payload,
    )
    assert isinstance(payload, dict)
    payload["example"] = "clients/openai_compatible_http_client.py"
    payload["trace"] = tracer.trace_info(request_id)
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
