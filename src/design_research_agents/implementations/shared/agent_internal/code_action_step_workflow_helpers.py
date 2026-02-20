"""Helper functions shared by workflow-native code action-step agents."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse


def assert_success_handler(context: Mapping[str, object]) -> Mapping[str, object]:
    """Fail workflow run when finalize emitted an unsuccessful agent result.

    Args:
        context: Workflow step execution context payload.

    Returns:
        Mapping confirming the assertion step completed.

    Raises:
        TypeError: If finalize output is missing the expected ``ExecutionResult`` payload.
        ValueError: If finalize output indicates an unsuccessful agent run.
    """
    finalize = dependency_output(context=context, step_id="finalize")
    agent_result = finalize.get("agent_result")
    if not isinstance(agent_result, ExecutionResult):
        raise TypeError("Finalize step did not provide an ExecutionResult payload.")
    if not agent_result.success:
        error_text = agent_result.output.get("error")
        raise ValueError(
            str(error_text)
            if isinstance(error_text, str) and error_text.strip()
            else "code action-step execution failed"
        )
    return {"ok": True}


def dependency_output(*, context: Mapping[str, object], step_id: str) -> dict[str, object]:
    """Extract one dependency step output payload from workflow step context.

    Args:
        context: Workflow step execution context payload.
        step_id: Dependency step identifier to retrieve.

    Returns:
        Normalized dependency output mapping, or an empty mapping when unavailable.
    """
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return {}
    step_result = dependency_results.get(step_id)
    if not isinstance(step_result, Mapping):
        return {}
    output = step_result.get("output")
    if isinstance(output, Mapping):
        return dict(output)
    return {}


def mapping_or_empty(value: object) -> dict[str, object]:
    """Normalize optional mapping payloads to plain dictionaries.

    Args:
        value: Candidate mapping value.

    Returns:
        Plain dictionary when ``value`` is a mapping, otherwise an empty dictionary.
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def int_or_default(value: object, *, default: int) -> int:
    """Return integer value when coercible; otherwise return the provided default.

    Args:
        value: Candidate integer-like payload.
        default: Fallback integer value when coercion fails.

    Returns:
        Parsed integer or fallback default value.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def llm_response_or_none(value: object) -> LLMResponse | None:
    """Return typed model response when available.

    Args:
        value: Candidate model response payload.

    Returns:
        Typed ``LLMResponse`` when available, otherwise ``None``.
    """
    return value if isinstance(value, LLMResponse) else None


__all__ = [
    "assert_success_handler",
    "dependency_output",
    "int_or_default",
    "llm_response_or_none",
    "mapping_or_empty",
]
