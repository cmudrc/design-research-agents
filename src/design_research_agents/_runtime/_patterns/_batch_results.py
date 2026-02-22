"""Delegate-batch result parsing helpers shared across pattern modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._llm import LLMResponse


def extract_delegate_batch_call_result_from_context(
    *,
    context: Mapping[str, object],
    dependency_step_id: str,
    call_id: str,
) -> Mapping[str, object] | None:
    """Extract one call-result mapping from a dependency batch-step output."""
    # Defensive shape checks keep pattern helpers resilient to partially populated context payloads.
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return None
    dependency_payload = dependency_results.get(dependency_step_id)
    if not isinstance(dependency_payload, Mapping):
        return None
    dependency_output = dependency_payload.get("output")
    if not isinstance(dependency_output, Mapping):
        return None
    results = dependency_output.get("results")
    if not isinstance(results, list):
        return None
    for result_entry in results:
        if not isinstance(result_entry, Mapping):
            continue
        if str(result_entry.get("call_id", "")) != call_id:
            continue
        # Return first matching id; call ids are expected to be unique within a batch.
        return result_entry
    return None


def extract_delegate_batch_call_result(
    *,
    results: Sequence[Mapping[str, object]],
    call_id: str,
) -> Mapping[str, object] | None:
    """Extract one call-result mapping from in-memory batch results."""
    for result_entry in results:
        if str(result_entry.get("call_id", "")) != call_id:
            continue
        return result_entry
    return None


def extract_call_output(call_result: Mapping[str, object] | None) -> dict[str, object]:
    """Extract normalized delegate output mapping from one call-result entry."""
    if not isinstance(call_result, Mapping):
        return {}
    output = call_result.get("output")
    if isinstance(output, Mapping):
        return dict(output)
    return {}


def extract_call_model_response(call_result: Mapping[str, object] | None) -> LLMResponse | None:
    """Deserialize model response payload from one call-result entry when present."""
    if not isinstance(call_result, Mapping):
        return None
    model_response = call_result.get("model_response")
    if isinstance(model_response, LLMResponse):
        return model_response
    if isinstance(model_response, Mapping):
        try:
            # Support both hydrated objects and serialized dict payloads from workflow outputs.
            return LLMResponse(**dict(model_response))
        except TypeError:
            return None
    return None


def is_call_success(call_result: Mapping[str, object] | None) -> bool:
    """Return whether one call-result entry succeeded."""
    if not isinstance(call_result, Mapping):
        return False
    return bool(call_result.get("success", False))


def extract_call_error(
    call_result: Mapping[str, object] | None,
    *,
    fallback_message: str,
) -> str:
    """Extract one human-readable error message from a call-result entry."""
    if not isinstance(call_result, Mapping):
        return fallback_message
    raw_error = call_result.get("error")
    if isinstance(raw_error, str) and raw_error.strip():
        return raw_error.strip()
    output = call_result.get("output")
    if isinstance(output, Mapping):
        output_error = output.get("error")
        if isinstance(output_error, str) and output_error.strip():
            return output_error.strip()
    return fallback_message


__all__ = [
    "extract_call_error",
    "extract_call_model_response",
    "extract_call_output",
    "extract_delegate_batch_call_result",
    "extract_delegate_batch_call_result_from_context",
    "is_call_success",
]
