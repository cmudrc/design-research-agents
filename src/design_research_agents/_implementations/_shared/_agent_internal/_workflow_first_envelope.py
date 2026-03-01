"""Helpers for assembling workflow-first agent output envelopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._execution import ExecutionResult


def build_workflow_first_output(
    *,
    base_output: Mapping[str, object] | None,
    workflow_result: ExecutionResult,
    final_output: object,
    artifacts: Sequence[object] | None = None,
) -> dict[str, object]:
    """Merge one agent payload into a canonical workflow-first output envelope.

    Args:
        base_output: Existing agent output fields to preserve.
        workflow_result: Workflow execution result that produced step metadata.
        final_output: Canonical final output payload for concise consumption.
        artifacts: Optional artifact manifest override.

    Returns:
        Output mapping with ``workflow``, ``final_output``, and ``artifacts`` keys.
    """
    merged_output = dict(base_output or {})
    workflow_payload = workflow_result.output.get("workflow")
    merged_output["workflow"] = (
        dict(workflow_payload)
        if isinstance(workflow_payload, Mapping)
        else workflow_result.to_dict().get("output", {}).get("workflow", {})
    )
    merged_output["final_output"] = final_output
    if artifacts is None:
        workflow_artifacts = workflow_result.output.get("artifacts")
        merged_output["artifacts"] = list(workflow_artifacts) if isinstance(workflow_artifacts, Sequence) else []
    else:
        merged_output["artifacts"] = list(artifacts)
    return merged_output


__all__ = [
    "build_workflow_first_output",
]
