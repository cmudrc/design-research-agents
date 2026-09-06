"""Execution-evidence normalization for agent paper contributions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from design_research_agents._contracts import ExecutionResult, ToolResult

_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_URL_PASSWORD_PATTERN = re.compile(r"(://[^:/\s]+:)[^@\s]+(@)")


def coerce_execution_result(result: ExecutionResult | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a typed or mapping execution result without carrying generated output."""
    if isinstance(result, ExecutionResult):
        return {
            "success": result.success,
            "model_response": result.model_response,
            "tool_results": list(result.tool_results),
            "step_results": dict(result.step_results),
            "execution_order": list(result.execution_order),
            "metadata": dict(result.metadata),
        }
    if not isinstance(result, Mapping):
        raise TypeError("execution_result must be an ExecutionResult or mapping.")
    raw_step_results = result.get("step_results")
    raw_metadata = result.get("metadata")
    return {
        "success": bool(result.get("success", False)),
        "model_response": result.get("model_response"),
        "tool_results": _sequence_or_empty(result.get("tool_results")),
        "step_results": dict(raw_step_results) if isinstance(raw_step_results, Mapping) else {},
        "execution_order": [str(item) for item in _sequence_or_empty(result.get("execution_order"))],
        "metadata": dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {},
    }


def resolve_evidence_refs(requested: Sequence[str], result: Mapping[str, Any]) -> list[str]:
    """Validate explicit evidence references and include a persisted trace path."""
    if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
        raise TypeError("evidence_refs must be a sequence of non-empty strings.")
    refs: list[str] = []
    for item in requested:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("evidence_refs must contain only non-empty strings.")
        refs.append(item.strip())
    metadata = result.get("metadata")
    trace_path = metadata.get("trace_path") if isinstance(metadata, Mapping) else None
    if isinstance(trace_path, str) and trace_path.strip():
        refs.append(trace_path.strip())
    return list(dict.fromkeys(refs))


def observed_contributions(
    result: Mapping[str, Any],
    *,
    component_id: str,
    evidence_refs: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build factual runtime contributions and gaps linked to durable evidence."""
    refs = list(evidence_refs)
    status = "succeeded" if result["success"] else "did not complete successfully"
    execution_order = [str(item) for item in result.get("execution_order", ())]
    step_statuses = _step_statuses(result.get("step_results", {}))
    contributions: list[dict[str, Any]] = [
        _observed_contribution(
            f"agent:{component_id}:execution",
            (
                f"The recorded execution {status}; it recorded {len(step_statuses)} step result(s)"
                f"{_execution_order_suffix(execution_order)}."
            ),
            evidence_refs=refs,
            metadata={
                "success": bool(result["success"]),
                "execution_order": execution_order,
                "step_statuses": step_statuses,
            },
        )
    ]
    gaps: list[dict[str, Any]] = []

    model_observation = _model_observation(result.get("model_response"))
    if model_observation is not None:
        contributions.append(
            _observed_contribution(
                f"agent:{component_id}:model",
                model_observation["text"],
                evidence_refs=refs,
                metadata=model_observation["metadata"],
            )
        )

    for index, raw_tool_result in enumerate(result.get("tool_results", ())):
        tool = _tool_result_snapshot(raw_tool_result)
        if tool is None:
            gaps.append(
                _gap(
                    component_id,
                    f"tool-result-{index + 1}-invalid",
                    f"Recorded tool result {index + 1} could not be normalized.",
                    evidence_refs=refs,
                )
            )
            continue
        outcome = "succeeded" if tool["ok"] else "failed"
        contributions.append(
            _observed_contribution(
                f"agent:{component_id}:tool:{index + 1}",
                f"The recorded execution invoked tool {tool['tool_name']!r}; the invocation {outcome}.",
                evidence_refs=refs,
                metadata=tool,
            )
        )
        if not tool["ok"]:
            gaps.append(
                _gap(
                    component_id,
                    f"tool-{index + 1}-failed",
                    f"Report and resolve the recorded failure of tool {tool['tool_name']!r}.",
                    evidence_refs=refs,
                )
            )

    metadata = result.get("metadata")
    trace_path = metadata.get("trace_path") if isinstance(metadata, Mapping) else None
    if isinstance(trace_path, str) and trace_path.strip():
        contributions.append(
            _observed_contribution(
                f"agent:{component_id}:trace",
                "The execution produced a persisted trace reference for audit and reproduction.",
                evidence_refs=refs,
                metadata={"trace_path": trace_path.strip()},
            )
        )
    if not result["success"]:
        gaps.append(
            _gap(
                component_id,
                "execution-unsuccessful",
                "The recorded execution did not complete successfully; report the failure before drawing conclusions.",
                evidence_refs=refs,
            )
        )
    return contributions, gaps


def has_trace_reference(
    execution_result: ExecutionResult | Mapping[str, Any] | None,
    evidence_refs: Sequence[str],
) -> bool:
    """Return whether explicit evidence or result metadata establishes a durable trace."""
    if evidence_refs:
        return True
    if isinstance(execution_result, ExecutionResult):
        trace_path = execution_result.metadata.get("trace_path")
    elif isinstance(execution_result, Mapping):
        metadata = execution_result.get("metadata")
        trace_path = metadata.get("trace_path") if isinstance(metadata, Mapping) else None
    else:
        trace_path = None
    return isinstance(trace_path, str) and bool(trace_path.strip())


def redact(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-compatible value with common credential shapes removed."""
    if key is not None and _is_secret_key(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(item_key): redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [redact(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return redact(asdict(value))
    return f"<{type(value).__name__}>"


def _observed_contribution(
    contribution_id: str,
    text: str,
    *,
    evidence_refs: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contribution_id": contribution_id,
        "section": "methods",
        "kind": "bullet",
        "text": text,
        "evidence_basis": "observed",
        "citation_keys": [],
        "evidence_refs": list(evidence_refs),
        "metadata": redact(metadata),
    }


def _step_statuses(raw_step_results: object) -> dict[str, str]:
    if not isinstance(raw_step_results, Mapping):
        return {}
    statuses: dict[str, str] = {}
    for raw_step_id, raw_result in raw_step_results.items():
        status = getattr(raw_result, "status", None)
        if status is None and isinstance(raw_result, Mapping):
            status = raw_result.get("status")
        statuses[str(raw_step_id)] = str(status or "unknown")
    return dict(sorted(statuses.items()))


def _execution_order_suffix(execution_order: Sequence[str]) -> str:
    if not execution_order:
        return ""
    return " in the persisted order " + ", ".join(repr(item) for item in execution_order)


def _model_observation(response: object) -> dict[str, Any] | None:
    if response is None:
        return None
    if isinstance(response, Mapping):
        model = response.get("model")
        provider = response.get("provider")
        finish_reason = response.get("finish_reason")
        latency_ms = response.get("latency_ms")
    else:
        model = getattr(response, "model", None)
        provider = getattr(response, "provider", None)
        finish_reason = getattr(response, "finish_reason", None)
        latency_ms = getattr(response, "latency_ms", None)
    provider_label = str(provider).strip() if provider is not None else "unreported provider"
    model_label = str(model).strip() if model is not None else "unreported model"
    response_source = f"{provider_label!r} using {model_label!r}"
    return {
        "text": f"The recorded execution received its final model response from {response_source}.",
        "metadata": {
            "provider": provider,
            "model": model,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
        },
    }


def _tool_result_snapshot(raw_result: object) -> dict[str, Any] | None:
    if isinstance(raw_result, ToolResult):
        return {
            "tool_name": raw_result.tool_name,
            "ok": raw_result.ok,
            "warning_count": len(raw_result.warnings),
            "artifact_count": len(raw_result.artifacts),
            "error_type": raw_result.error.type if raw_result.error is not None else None,
        }
    if not isinstance(raw_result, Mapping):
        return None
    tool_name = raw_result.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    error = raw_result.get("error")
    error_type = error.get("type") if isinstance(error, Mapping) else None
    return {
        "tool_name": tool_name.strip(),
        "ok": bool(raw_result.get("ok", False)),
        "warning_count": len(_sequence_or_empty(raw_result.get("warnings"))),
        "artifact_count": len(_sequence_or_empty(raw_result.get("artifacts"))),
        "error_type": error_type,
    }


def _sequence_or_empty(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _gap(
    component_id: str,
    suffix: str,
    message: str,
    *,
    evidence_refs: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "gap_id": f"agent:{component_id}:{suffix}",
        "section": "methods",
        "message": message,
        "evidence_refs": list(evidence_refs),
    }


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _redact_string(value: str) -> str:
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    redacted = _OPENAI_KEY_PATTERN.sub("[REDACTED]", redacted)
    return _URL_PASSWORD_PATTERN.sub(r"\1[REDACTED]\2", redacted)


__all__ = [
    "coerce_execution_result",
    "has_trace_reference",
    "observed_contributions",
    "redact",
    "resolve_evidence_refs",
]
