"""Simple trace-analysis harness for JSONL trace artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def analyze_trace_dir(
    *,
    trace_dir: Path,
    glob_pattern: str = "run_*.jsonl",
    top_n: int = 10,
) -> dict[str, object]:
    """Analyze trace JSONL files under one directory."""
    trace_files = sorted(trace_dir.glob(glob_pattern))
    events: list[dict[str, object]] = []
    malformed_lines = 0
    for trace_path in trace_files:
        file_events, file_malformed = _read_trace_file(trace_path)
        events.extend(file_events)
        malformed_lines += file_malformed

    event_counts = Counter(_event_type(event) for event in events)
    # Prefer newer "*Observed" events when available; fall back to legacy names for compatibility.
    has_model_request_observed = event_counts.get("ModelRequestObserved", 0) > 0
    has_model_response_observed = event_counts.get("ModelResponseObserved", 0) > 0
    has_tool_observed = event_counts.get("ToolInvocationObserved", 0) > 0
    has_tool_result_observed = event_counts.get("ToolResultObserved", 0) > 0
    has_step_result_observed = event_counts.get("WorkflowStepResultObserved", 0) > 0

    run_outcomes = _resolve_run_outcomes(events)
    model_metrics = _build_model_metrics(
        events,
        prefer_request_observed=has_model_request_observed,
        prefer_response_observed=has_model_response_observed,
        top_n=top_n,
    )
    tool_metrics = _build_tool_metrics(
        events,
        prefer_observed=has_tool_observed,
        prefer_result_observed=has_tool_result_observed,
        top_n=top_n,
    )
    workflow_metrics = _build_workflow_step_metrics(events, prefer_observed=has_step_result_observed)
    latency_metrics = _build_latency_metrics(events, top_n=top_n)
    error_metrics = _build_error_metrics(events, top_n=top_n)

    return {
        "inputs": {
            "trace_dir": str(trace_dir),
            "glob": glob_pattern,
            "top_n": top_n,
            "files_scanned": len(trace_files),
            "files": [str(path) for path in trace_files],
        },
        "runs": run_outcomes,
        "events": {
            "total": len(events),
            "counts_by_type": dict(sorted(event_counts.items())),
            "malformed_lines": malformed_lines,
        },
        "models": model_metrics,
        "tools": tool_metrics,
        "workflow_steps": workflow_metrics,
        "latency": latency_metrics,
        "errors": error_metrics,
        "warnings": {
            "malformed_lines": malformed_lines,
        },
    }


def _read_trace_file(path: Path) -> tuple[list[dict[str, object]], int]:
    """Read one JSONL trace file into event dictionaries."""
    parsed_events: list[dict[str, object]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                # Count malformed lines but continue so one bad line doesn't drop the whole file.
                malformed += 1
                continue
            if not isinstance(payload, dict):
                malformed += 1
                continue
            parsed_events.append(payload)
    return parsed_events, malformed


def _resolve_run_outcomes(events: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Resolve run-level success/failure summaries."""
    run_outcomes: dict[str, bool] = {}
    fallback_agent_outcomes: dict[str, bool] = {}
    for event in events:
        run_id = str(event.get("run_id", "") or "")
        if not run_id:
            continue
        event_type = _event_type(event)
        attributes = _event_attributes(event)
        success = _coerce_bool(attributes.get("success"))
        if success is None:
            continue
        if event_type == "RunFinished":
            run_outcomes[run_id] = success
        elif event_type == "AgentRunFinished":
            fallback_agent_outcomes[run_id] = success

    for run_id, success in fallback_agent_outcomes.items():
        # Preserve explicit RunFinished outcomes when both event families exist.
        run_outcomes.setdefault(run_id, success)

    success_count = sum(1 for value in run_outcomes.values() if value)
    failure_count = sum(1 for value in run_outcomes.values() if not value)
    return {
        "unique_run_count": len({str(event.get("run_id", "") or "") for event in events if event.get("run_id")}),
        "runs_with_terminal_status": len(run_outcomes),
        "success_count": success_count,
        "failure_count": failure_count,
    }


def _build_model_metrics(
    events: Sequence[Mapping[str, object]],
    *,
    prefer_request_observed: bool,
    prefer_response_observed: bool,
    top_n: int,
) -> dict[str, object]:
    """Build model usage metrics from trace events."""
    call_event_types = ("ModelRequestObserved",) if prefer_request_observed else ("ModelCallStarted",)
    response_event_types = ("ModelResponseObserved",) if prefer_response_observed else ("ModelCallFinished",)
    failure_event_types = ("ModelResponseObserved",) if prefer_response_observed else ("ModelCallFailed",)

    model_call_count = 0
    per_model_calls: Counter[str] = Counter()
    for event in events:
        if _event_type(event) not in call_event_types:
            continue
        model_call_count += 1
        model_name = _resolve_model_name(event)
        per_model_calls[model_name] += 1

    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    per_model_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    model_latency_values: list[int] = []
    model_failures = 0
    for event in events:
        event_type = _event_type(event)
        if event_type in response_event_types:
            model_name = _resolve_model_name(event)
            usage = _extract_usage(event)
            for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = _coerce_int(usage.get(token_key))
                if value is None:
                    continue
                tokens[token_key] += value
                per_model_tokens[model_name][token_key] += value
            response_latency = _extract_response_latency(event)
            if response_latency is not None:
                model_latency_values.append(response_latency)
        if event_type in failure_event_types and _extract_error_text(event):
            model_failures += 1

    per_model_items = [
        {
            "model": model_name,
            "calls": per_model_calls.get(model_name, 0),
            "prompt_tokens": int(per_model_tokens.get(model_name, Counter()).get("prompt_tokens", 0)),
            "completion_tokens": int(per_model_tokens.get(model_name, Counter()).get("completion_tokens", 0)),
            "total_tokens": int(per_model_tokens.get(model_name, Counter()).get("total_tokens", 0)),
        }
        for model_name in sorted(set(per_model_calls) | set(per_model_tokens))
    ]
    per_model_items.sort(key=lambda item: _coerce_int(item.get("calls")) or 0, reverse=True)

    return {
        "model_call_count": model_call_count,
        "model_failure_count": model_failures,
        "tokens": tokens,
        "per_model": per_model_items[: max(1, top_n)],
        "latency_ms": _summarize_numbers(model_latency_values),
    }


def _build_tool_metrics(
    events: Sequence[Mapping[str, object]],
    *,
    prefer_observed: bool,
    prefer_result_observed: bool,
    top_n: int,
) -> dict[str, object]:
    """Build tool usage metrics from trace events."""
    invocation_event_types = ("ToolInvocationObserved",) if prefer_observed else ("ToolCallStarted",)
    result_event_types = ("ToolResultObserved",) if prefer_result_observed else ("ToolCallFinished", "ToolCallFailed")
    # Keep invocation/result event families configurable to support mixed-version trace directories.

    tool_invocation_count = 0
    per_tool_counts: Counter[str] = Counter()
    for event in events:
        if _event_type(event) not in invocation_event_types:
            continue
        tool_name = _resolve_tool_name(event)
        tool_invocation_count += 1
        per_tool_counts[tool_name] += 1

    tool_success = 0
    tool_failures = 0
    per_tool_failures: Counter[str] = Counter()
    for event in events:
        if _event_type(event) not in result_event_types:
            continue
        tool_name = _resolve_tool_name(event)
        ok_flag = _extract_tool_success(event)
        if ok_flag is True:
            tool_success += 1
        elif ok_flag is False:
            tool_failures += 1
            per_tool_failures[tool_name] += 1

    per_tool = [
        {
            "tool_name": tool_name,
            "invocations": per_tool_counts.get(tool_name, 0),
            "failures": per_tool_failures.get(tool_name, 0),
        }
        for tool_name in sorted(set(per_tool_counts) | set(per_tool_failures))
    ]
    per_tool.sort(key=lambda item: _coerce_int(item.get("invocations")) or 0, reverse=True)

    return {
        "tool_invocation_count": tool_invocation_count,
        "tool_success_count": tool_success,
        "tool_failure_count": tool_failures,
        "per_tool": per_tool[: max(1, top_n)],
    }


def _build_workflow_step_metrics(
    events: Sequence[Mapping[str, object]],
    *,
    prefer_observed: bool,
) -> dict[str, object]:
    """Build workflow step metrics from trace events."""
    result_event_types = ("WorkflowStepResultObserved",) if prefer_observed else ("WorkflowStepFinished",)

    step_count = 0
    counts_by_type: Counter[str] = Counter()
    counts_by_status: Counter[str] = Counter()
    for event in events:
        if _event_type(event) not in result_event_types:
            continue
        attributes = _event_attributes(event)
        step_count += 1
        counts_by_type[str(attributes.get("step_type", "unknown"))] += 1
        counts_by_status[str(attributes.get("status", "unknown"))] += 1

    return {
        "step_execution_count": step_count,
        "counts_by_step_type": dict(sorted(counts_by_type.items())),
        "counts_by_status": dict(sorted(counts_by_status.items())),
    }


def _build_latency_metrics(events: Sequence[Mapping[str, object]], *, top_n: int) -> dict[str, object]:
    """Build latency summaries from event duration and response fields."""
    tool_durations: list[int] = []
    duration_by_event_type: dict[str, list[int]] = defaultdict(list)
    for event in events:
        duration_ms = _coerce_int(event.get("duration_ms"))
        if duration_ms is None:
            continue
        event_type = _event_type(event)
        duration_by_event_type[event_type].append(duration_ms)
        if event_type in {"ToolCallFinished", "ToolCallFailed"}:
            tool_durations.append(duration_ms)

    key_event_durations = [
        {
            "event_type": event_type,
            "summary": _summarize_numbers(values),
        }
        for event_type, values in duration_by_event_type.items()
    ]
    key_event_durations.sort(
        key=lambda item: _coerce_int(_as_mapping(item.get("summary")).get("count")) or 0,
        reverse=True,
    )

    return {
        "tool_duration_ms": _summarize_numbers(tool_durations),
        "event_duration_ms": key_event_durations[: max(1, top_n)],
    }


def _build_error_metrics(events: Sequence[Mapping[str, object]], *, top_n: int) -> dict[str, object]:
    """Build error summaries from error-bearing events."""
    error_counter: Counter[str] = Counter()
    for event in events:
        error_text = _extract_error_text(event)
        if error_text:
            error_counter[error_text] += 1
    top_errors = [{"error": error, "count": count} for error, count in error_counter.most_common(max(1, top_n))]
    return {"error_event_count": sum(error_counter.values()), "top_errors": top_errors}


def _event_type(event: Mapping[str, object]) -> str:
    """Return normalized event type."""
    return str(event.get("event_type", "Unknown"))


def _event_attributes(event: Mapping[str, object]) -> Mapping[str, object]:
    """Return event attributes payload as mapping."""
    attributes = event.get("attributes")
    if not isinstance(attributes, Mapping):
        return {}
    return cast_mapping(attributes)


def _resolve_model_name(event: Mapping[str, object]) -> str:
    """Resolve one model name from event attributes."""
    attributes = _event_attributes(event)
    explicit_model = attributes.get("model")
    if isinstance(explicit_model, str) and explicit_model:
        return explicit_model
    response = attributes.get("response")
    if isinstance(response, Mapping):
        response_model = response.get("model")
        if isinstance(response_model, str) and response_model:
            return response_model
    request_payload = attributes.get("request")
    if isinstance(request_payload, Mapping):
        request_model = request_payload.get("model")
        if isinstance(request_model, str) and request_model:
            return request_model
    return "unknown"


def _resolve_tool_name(event: Mapping[str, object]) -> str:
    """Resolve one tool name from event attributes."""
    attributes = _event_attributes(event)
    tool_name = attributes.get("tool_name")
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    return "unknown"


def _extract_usage(event: Mapping[str, object]) -> Mapping[str, object]:
    """Extract usage payload from event attributes."""
    attributes = _event_attributes(event)
    usage_direct = attributes.get("usage")
    if isinstance(usage_direct, Mapping):
        return cast_mapping(usage_direct)
    response_payload = attributes.get("response")
    if isinstance(response_payload, Mapping):
        response_usage = response_payload.get("usage")
        if isinstance(response_usage, Mapping):
            return cast_mapping(response_usage)
    return {}


def _extract_response_latency(event: Mapping[str, object]) -> int | None:
    """Extract latency field from response payload when present."""
    attributes = _event_attributes(event)
    response_payload = attributes.get("response")
    if isinstance(response_payload, Mapping):
        latency_ms = _coerce_int(response_payload.get("latency_ms"))
        if latency_ms is not None:
            return latency_ms
    return _coerce_int(event.get("duration_ms"))


def _extract_tool_success(event: Mapping[str, object]) -> bool | None:
    """Infer success flag from a tool result event."""
    event_type = _event_type(event)
    attributes = _event_attributes(event)
    if event_type == "ToolResultObserved":
        explicit_ok = _coerce_bool(attributes.get("ok"))
        if explicit_ok is not None:
            return explicit_ok
        # When explicit ok is absent, infer failure from any error payload presence.
        return _extract_error_text(event) is None
    if event_type == "ToolCallFinished":
        return True
    if event_type == "ToolCallFailed":
        return False
    return None


def _extract_error_text(event: Mapping[str, object]) -> str | None:
    """Extract one error message from an event payload when available."""
    attributes = _event_attributes(event)
    error_value = attributes.get("error")
    if isinstance(error_value, str) and error_value:
        return error_value
    if isinstance(error_value, Mapping):
        message = error_value.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(dict(error_value), ensure_ascii=True, sort_keys=True)
    if isinstance(error_value, list) and error_value:
        return json.dumps(error_value, ensure_ascii=True, sort_keys=True)
    return None


def _summarize_numbers(values: Sequence[int]) -> dict[str, object]:
    """Return count/min/p50/p95/max summary for integer values."""
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    # Use nearest-rank percentiles to keep summaries integer-valued and deterministic.
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(sorted_values: Sequence[int], quantile: float) -> int:
    """Return nearest-rank percentile for pre-sorted integer values."""
    if not sorted_values:
        return 0
    rank = round((len(sorted_values) - 1) * quantile)
    # Clamp for safety in case callers pass out-of-range quantiles.
    rank = max(0, min(rank, len(sorted_values) - 1))
    return int(sorted_values[rank])


def _coerce_int(value: object) -> int | None:
    """Coerce value to int when safely possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _coerce_bool(value: object) -> bool | None:
    """Coerce value to bool when safely possible."""
    if isinstance(value, bool):
        return value
    return None


def cast_mapping(value: Mapping[Any, Any]) -> Mapping[str, object]:
    """Cast a generic mapping into string-keyed object mapping."""
    return {str(key): val for key, val in value.items()}


def _as_mapping(value: object) -> Mapping[str, object]:
    """Return value as mapping when possible, else empty mapping."""
    if isinstance(value, Mapping):
        return cast_mapping(value)
    return {}


def _format_human_summary(payload: Mapping[str, object]) -> str:
    """Render human-readable summary for analysis output."""
    runs = cast_mapping(_mapping_value(payload, "runs"))
    events = cast_mapping(_mapping_value(payload, "events"))
    models = cast_mapping(_mapping_value(payload, "models"))
    tools = cast_mapping(_mapping_value(payload, "tools"))
    workflow_steps = cast_mapping(_mapping_value(payload, "workflow_steps"))
    latency = cast_mapping(_mapping_value(payload, "latency"))
    errors = cast_mapping(_mapping_value(payload, "errors"))
    warnings = cast_mapping(_mapping_value(payload, "warnings"))
    inputs = cast_mapping(_mapping_value(payload, "inputs"))

    lines = [
        "Trace Analysis Summary",
        f"- trace_dir: {inputs.get('trace_dir')}",
        f"- files_scanned: {inputs.get('files_scanned')}",
        f"- events_total: {events.get('total')}",
        f"- runs_unique: {runs.get('unique_run_count')}",
        f"- runs_success: {runs.get('success_count')}",
        f"- runs_failure: {runs.get('failure_count')}",
        "",
        "Model Usage",
        f"- model_call_count: {models.get('model_call_count')}",
        f"- model_failure_count: {models.get('model_failure_count')}",
        f"- prompt_tokens: {cast_mapping(_mapping_value(models, 'tokens')).get('prompt_tokens')}",
        f"- completion_tokens: {cast_mapping(_mapping_value(models, 'tokens')).get('completion_tokens')}",
        f"- total_tokens: {cast_mapping(_mapping_value(models, 'tokens')).get('total_tokens')}",
        "",
        "Tool Usage",
        f"- tool_invocation_count: {tools.get('tool_invocation_count')}",
        f"- tool_success_count: {tools.get('tool_success_count')}",
        f"- tool_failure_count: {tools.get('tool_failure_count')}",
        "",
        "Workflow Steps",
        f"- step_execution_count: {workflow_steps.get('step_execution_count')}",
        f"- counts_by_status: {json.dumps(_mapping_value(workflow_steps, 'counts_by_status'), sort_keys=True)}",
        "",
        "Latency",
        f"- model_latency_ms: {json.dumps(_mapping_value(models, 'latency_ms'), sort_keys=True)}",
        f"- tool_duration_ms: {json.dumps(_mapping_value(latency, 'tool_duration_ms'), sort_keys=True)}",
        "",
        "Errors",
        f"- error_event_count: {errors.get('error_event_count')}",
        f"- malformed_lines: {warnings.get('malformed_lines')}",
    ]
    return "\n".join(lines)


def _mapping_value(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Return mapping value from payload by key with empty fallback."""
    value = payload.get(key, {})
    if isinstance(value, Mapping):
        return cast_mapping(value)
    return {}


def main(argv: Iterable[str] | None = None) -> int:
    """Run trace-analysis CLI."""
    parser = argparse.ArgumentParser(description="Analyze trace JSONL files and compute summary metrics.")
    parser.add_argument("--trace-dir", required=True, help="Directory containing trace JSONL files.")
    parser.add_argument("--glob", default="run_*.jsonl", help="Glob pattern within trace directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output instead of human summary.")
    parser.add_argument("--top-n", type=int, default=10, help="Top N rows for per-model/per-tool summaries.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = analyze_trace_dir(
        trace_dir=Path(args.trace_dir),
        glob_pattern=str(args.glob),
        top_n=max(1, int(args.top_n)),
    )
    try:
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        print(_format_human_summary(summary))
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
