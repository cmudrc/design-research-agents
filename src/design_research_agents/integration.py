"""Study-facing integration helpers for executable agents."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

type AgentBinding = Any | Callable[[Any], Any]

_AGENT_EXECUTION_PARAMETER_NAMES = frozenset(
    {
        "prompt",
        "input",
        "request_id",
        "dependencies",
    }
)


@dataclass(slots=True)
class AgentExecutionEnvelope:
    """Normalized execution envelope used by study-orchestration consumers."""

    output: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    trace_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def execute_agent_run(
    agent_ref: Any,
    *,
    prompt: Any,
    request_id: str | None,
    dependencies: Mapping[str, object] | None,
    agent_bindings: Mapping[str, AgentBinding] | None = None,
) -> AgentExecutionEnvelope:
    """Execute one public agent reference through the stable prompt/dependencies contract."""
    resolved_dependencies = dict(dependencies or {})
    condition = resolved_dependencies.get("condition")
    executable = _resolve_agent_ref(
        agent_ref,
        condition=condition,
        agent_bindings=agent_bindings,
    )
    raw = _invoke_agent(
        executable=executable,
        prompt=prompt,
        request_id=request_id,
        dependencies=resolved_dependencies,
    )
    return _normalize_agent_execution(raw=raw, request_id=request_id)


def _resolve_agent_ref(
    agent_ref: Any,
    *,
    condition: Any,
    agent_bindings: Mapping[str, AgentBinding] | None,
) -> Any:
    """Resolve one agent reference into an executable object."""
    if not isinstance(agent_ref, str):
        return agent_ref

    if agent_bindings and agent_ref in agent_bindings:
        return _materialize_agent_binding(agent_bindings[agent_ref], condition)

    exported = getattr(import_module("design_research_agents"), agent_ref, None)
    if exported is None:
        raise ValueError(
            f"Unknown agent reference '{agent_ref}'. Register an agent binding or use a public export."
        )

    if inspect.isclass(exported):
        return exported()
    return exported


def _materialize_agent_binding(binding: AgentBinding, condition: Any) -> Any:
    """Resolve one binding into a concrete executable for the current condition."""
    if _is_condition_scoped_binding(binding):
        return binding(condition)
    return binding


def _is_condition_scoped_binding(binding: AgentBinding) -> bool:
    """Return whether one binding is a condition-to-executable builder."""
    if not callable(binding):
        return False
    if hasattr(binding, "run") and callable(binding.run):
        return False

    try:
        parameters = inspect.signature(binding).parameters
    except (TypeError, ValueError):
        return False

    if any(name in _AGENT_EXECUTION_PARAMETER_NAMES for name in parameters):
        return False

    if any(
        parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters.values()
    ):
        return False

    positional_parameters = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(parameters) == 1 and len(positional_parameters) == 1


def _invoke_agent(
    *,
    executable: Any,
    prompt: Any,
    request_id: str | None,
    dependencies: Mapping[str, object],
) -> Any:
    """Invoke one agent object or callable through the public prompt/dependencies contract."""
    if hasattr(executable, "run") and callable(executable.run):
        return _invoke_callable(
            callable_obj=executable.run,
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )

    if callable(executable):
        return _invoke_callable(
            callable_obj=executable,
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )

    raise ValueError("Resolved agent object is not executable.")


def _invoke_callable(
    *,
    callable_obj: Callable[..., Any],
    prompt: Any,
    request_id: str | None,
    dependencies: Mapping[str, object],
) -> Any:
    """Invoke one callable using the stable agent-entrypoint contract."""
    parameters = inspect.signature(callable_obj).parameters
    kwargs: dict[str, Any] = {}

    if "prompt" in parameters and parameters["prompt"].kind is not inspect.Parameter.POSITIONAL_ONLY:
        kwargs["prompt"] = prompt
    if "input" in parameters and parameters["input"].kind is not inspect.Parameter.POSITIONAL_ONLY:
        kwargs["input"] = prompt
    if "request_id" in parameters:
        kwargs["request_id"] = request_id
    if "dependencies" in parameters:
        kwargs["dependencies"] = dependencies

    if kwargs:
        return callable_obj(**kwargs)

    if not parameters:
        return callable_obj()

    positional_parameters = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional_parameters) == 1:
        return callable_obj(prompt)

    raise ValueError(
        "Agent execution target must accept the public prompt/request_id/dependencies "
        "contract or a single prompt argument."
    )


def _normalize_agent_execution(
    *,
    raw: Any,
    request_id: str | None,
) -> AgentExecutionEnvelope:
    """Normalize raw execution output to the canonical study-facing envelope."""
    if _is_execution_result(raw):
        return _normalize_execution_result(raw=raw, request_id=request_id)

    if isinstance(raw, Mapping):
        output = dict(raw.get("output", raw.get("outputs", {})))
        if not output and "text" in raw:
            output = {"text": raw["text"]}

        metrics = dict(raw.get("metrics", {}))
        trace_refs = [str(value) for value in raw.get("trace_refs", [])]
        metadata = dict(raw.get("metadata", {}))
        events = _normalize_events(
            raw_events=raw.get("events", []),
            request_id=request_id,
            output=output,
        )
        return AgentExecutionEnvelope(
            output=output,
            metrics=metrics,
            events=events,
            trace_refs=trace_refs,
            metadata=metadata,
        )

    output = {"text": str(raw)}
    return AgentExecutionEnvelope(
        output=output,
        metrics={},
        events=_normalize_events(raw_events=[], request_id=request_id, output=output),
        trace_refs=[],
        metadata={},
    )


def _is_execution_result(raw: Any) -> bool:
    """Return whether one object looks like a design-research-agents `ExecutionResult`."""
    if raw is None or isinstance(raw, Mapping):
        return False
    return all(hasattr(raw, attribute) for attribute in ("success", "output", "metadata"))


def _normalize_execution_result(
    *,
    raw: Any,
    request_id: str | None,
) -> AgentExecutionEnvelope:
    """Normalize a design-research-agents `ExecutionResult` into the study-facing envelope."""
    output_mapping = _extract_output_mapping(getattr(raw, "output", {}))
    output = _extract_execution_output(output_mapping)
    raw_metadata = getattr(raw, "metadata", {})
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    model_response = getattr(raw, "model_response", None)
    metadata.update(_extract_model_metadata(model_response))

    metrics: dict[str, Any] = {}
    raw_metrics = output_mapping.get("metrics")
    if isinstance(raw_metrics, Mapping):
        metrics.update(raw_metrics)
    _merge_usage_metrics(metrics, model_response)

    trace_refs = _extract_trace_refs(metadata)
    raw_events = output_mapping.get("events", [])
    normalized_raw_events = (
        raw_events
        if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes))
        else []
    )
    events = _normalize_events(
        raw_events=normalized_raw_events,
        request_id=request_id,
        output=output,
    )
    return AgentExecutionEnvelope(
        output=output,
        metrics=metrics,
        events=events,
        trace_refs=trace_refs,
        metadata=metadata,
    )


def _extract_output_mapping(raw_output: Any) -> dict[str, Any]:
    """Normalize the raw execution output envelope to a plain mapping."""
    if isinstance(raw_output, Mapping):
        return dict(raw_output)
    return {}


def _extract_execution_output(output_mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the canonical output payload from a workflow-style envelope."""
    final_output = output_mapping.get("final_output")
    if isinstance(final_output, Mapping):
        return dict(final_output)
    if isinstance(final_output, str):
        return {"text": final_output}
    if final_output is not None:
        return {"final_output": final_output}

    if "text" in output_mapping:
        return {"text": str(output_mapping["text"])}
    if "model_text" in output_mapping:
        return {"text": str(output_mapping["model_text"])}
    return dict(output_mapping)


def _merge_usage_metrics(metrics: dict[str, Any], model_response: Any) -> None:
    """Merge token-usage metadata from an optional model response."""
    usage = getattr(model_response, "usage", None)
    if isinstance(usage, Mapping):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    if isinstance(prompt_tokens, int):
        metrics.setdefault("input_tokens", prompt_tokens)
    if isinstance(completion_tokens, int):
        metrics.setdefault("output_tokens", completion_tokens)
    if isinstance(total_tokens, int):
        metrics.setdefault("total_tokens", total_tokens)


def _extract_model_metadata(model_response: Any) -> dict[str, Any]:
    """Extract stable model metadata from an optional LLM response."""
    metadata: dict[str, Any] = {}
    model_name = getattr(model_response, "model", None)
    if isinstance(model_name, str) and model_name.strip():
        metadata["model_name"] = model_name
    model_provider = getattr(model_response, "provider", None)
    if isinstance(model_provider, str) and model_provider.strip():
        metadata["model_provider"] = model_provider
    return metadata


def _extract_trace_refs(metadata: Mapping[str, Any]) -> list[str]:
    """Extract canonical trace references from execution metadata."""
    trace_refs: list[str] = []
    trace_path = metadata.get("trace_path")
    if isinstance(trace_path, str) and trace_path.strip():
        trace_refs.append(trace_path)
    raw_trace_refs = metadata.get("trace_refs")
    if isinstance(raw_trace_refs, Sequence) and not isinstance(raw_trace_refs, (str, bytes)):
        for value in raw_trace_refs:
            if isinstance(value, str) and value.strip() and value not in trace_refs:
                trace_refs.append(value)
    return trace_refs


def _normalize_events(
    *,
    raw_events: Sequence[Any],
    request_id: str | None,
    output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize raw event payloads into canonical event dictionaries."""
    events: list[dict[str, Any]] = []

    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            continue
        meta_json = raw_event.get("meta_json", {})
        if not isinstance(meta_json, Mapping):
            meta_json = {"value": meta_json}
        event_payload: dict[str, Any] = {
            "timestamp": str(raw_event.get("timestamp", _utc_now_iso())),
            "record_id": str(
                raw_event.get(
                    "record_id",
                    _hash_identifier(
                        "evt",
                        {
                            "request_id": request_id,
                            "index": index,
                            "event_type": raw_event.get("event_type", "event"),
                        },
                    ),
                )
            ),
            "text": str(raw_event.get("text", "")),
            "session_id": str(raw_event.get("session_id", request_id or "")),
            "actor_id": str(raw_event.get("actor_id", "agent")),
            "event_type": str(raw_event.get("event_type", "event")),
            "meta_json": dict(meta_json),
        }
        for key in ("level", "trial_id", "step_id", "tool_name", "evaluation_id", "run_id"):
            if key in raw_event:
                event_payload[key] = raw_event[key]
        events.append(event_payload)

    if events:
        return events

    return [
        {
            "timestamp": _utc_now_iso(),
            "record_id": _hash_identifier("evt", {"request_id": request_id, "index": 0}),
            "text": str(output.get("text", "")),
            "session_id": str(request_id or ""),
            "actor_id": "agent",
            "event_type": "assistant_output",
            "meta_json": {"auto_generated": True},
            "level": "step",
        }
    ]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _hash_identifier(prefix: str, payload: Mapping[str, Any]) -> str:
    """Build one deterministic identifier for normalized event records."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    return f"{prefix}-{digest}"


__all__ = ["AgentExecutionEnvelope", "execute_agent_run"]
