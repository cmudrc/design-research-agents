"""Reusable helpers for design-focused runnable examples."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from design_research_agents._contracts import LLMChatParams, LLMClient, LLMMessage, LLMResponse
from design_research_agents._shared._deterministic_design_helpers import (
    DeterministicSequenceLLMClient,
)
from design_research_agents._tracing import Tracer
from design_research_agents._tracing._context import finish_trace_run, start_trace_run

TRACE_DIR = Path("artifacts/examples/traces")


@dataclass(slots=True, frozen=True)
class ExampleTraceInfo:
    """Trace artifact metadata for one example run."""

    request_id: str
    trace_path: str | None
    trace_dir: str


def _sanitize_request_id(value: str) -> str:
    """Return a trace-file-safe request id representation."""
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe or "example_run"


def make_tracer() -> Tracer:
    """Return a tracer configured for deterministic example artifacts."""
    return Tracer(
        enabled=True,
        trace_dir=TRACE_DIR,
        enable_jsonl=True,
        enable_console=False,
    )


def resolve_trace_path(request_id: str, *, trace_dir: Path = TRACE_DIR) -> str | None:
    """Resolve latest JSONL trace path for one request id."""
    safe_request_id = _sanitize_request_id(request_id)
    candidates = sorted(
        trace_dir.glob(f"run_*_{safe_request_id}.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return str(candidates[0])


def trace_info(request_id: str, *, trace_dir: Path = TRACE_DIR) -> dict[str, object]:
    """Return a JSON-serializable trace metadata payload."""
    return asdict(
        ExampleTraceInfo(
            request_id=request_id,
            trace_path=resolve_trace_path(request_id, trace_dir=trace_dir),
            trace_dir=str(trace_dir),
        )
    )


def print_json(payload: dict[str, object]) -> None:
    """Print stable JSON output for deterministic example assertions."""
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


def run_representative_chat(
    *,
    client: LLMClient,
    prompt: str,
    deterministic_response: str,
    system_prompt: str = "You are a concise engineering design assistant.",
    max_tokens: int = 120,
) -> dict[str, object]:
    """Run one representative chat completion and return normalized call metadata."""
    deterministic_mode = os.environ.get("DRA_EXAMPLE_LLM_MODE", "").strip().lower() == "deterministic"
    model = client.default_model()
    call_client: LLMClient = (
        cast(LLMClient, DeterministicSequenceLLMClient(responses=(deterministic_response,)))
        if deterministic_mode
        else client
    )
    response = call_client.chat(
        [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=prompt),
        ],
        model=model,
        params=LLMChatParams(temperature=0.0, max_tokens=max_tokens),
    )
    if not isinstance(response, LLMResponse):
        raise TypeError("Client chat response must be an LLMResponse instance.")
    return {
        "execution_mode": "deterministic_stub" if deterministic_mode else "live_client",
        "prompt": prompt,
        "response_text": response.text,
        "response_model": response.model,
        "response_provider": response.provider,
        "response_has_text": bool(response.text.strip()),
    }


def run_traced_callable(
    *,
    agent_name: str,
    request_id: str,
    input_payload: dict[str, object],
    function: Callable[[], object],
) -> object:
    """Run one callable wrapped in explicit trace session lifecycle."""
    tracer = make_tracer()
    scope = start_trace_run(
        agent_name=agent_name,
        request_id=request_id,
        input_payload=input_payload,
        dependencies={},
        tracer=tracer,
    )
    try:
        value = function()
    except Exception as exc:
        finish_trace_run(scope, error=str(exc))
        raise
    finish_trace_run(scope, result=SimpleNamespace(success=True, output=value))
    return value


__all__ = [
    "TRACE_DIR",
    "make_tracer",
    "print_json",
    "resolve_trace_path",
    "run_representative_chat",
    "run_traced_callable",
    "trace_info",
]
