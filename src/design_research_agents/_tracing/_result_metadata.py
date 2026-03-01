"""Trace metadata helpers for execution results."""

from __future__ import annotations

from design_research_agents._contracts._execution import ExecutionResult

from ._config import Tracer


def enrich_execution_result_trace_metadata(
    *,
    result: ExecutionResult,
    tracer: Tracer | None,
) -> ExecutionResult:
    """Attach tracer-derived trace metadata to one execution result.

    Args:
        result: Execution result payload to enrich.
        tracer: Optional tracer used to resolve trace file metadata.

    Returns:
        Result with ``request_id``/``trace_dir``/``trace_path`` metadata when available.
    """
    if tracer is None or not tracer.enabled:
        return result

    request_id_raw = result.metadata.get("request_id")
    if not isinstance(request_id_raw, str):
        return result
    request_id = request_id_raw.strip()
    if not request_id:
        return result

    trace_info = tracer.trace_info(request_id)
    merged_metadata = dict(result.metadata)
    merged_metadata["request_id"] = request_id
    trace_dir = trace_info.get("trace_dir")
    if isinstance(trace_dir, str) and trace_dir.strip():
        merged_metadata["trace_dir"] = trace_dir
    trace_path = trace_info.get("trace_path")
    if isinstance(trace_path, str) and trace_path.strip():
        merged_metadata["trace_path"] = trace_path

    if merged_metadata == result.metadata:
        return result

    return ExecutionResult(
        success=result.success,
        output=dict(result.output),
        tool_results=list(result.tool_results),
        model_response=result.model_response,
        step_results=dict(result.step_results),
        execution_order=list(result.execution_order),
        metadata=merged_metadata,
    )


__all__ = ["enrich_execution_result_trace_metadata"]
