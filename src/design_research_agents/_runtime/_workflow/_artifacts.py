"""Artifact helpers shared by workflow runtime engines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._workflow import WorkflowArtifact, WorkflowStepResult


def dedupe_artifacts(artifacts: Sequence[WorkflowArtifact]) -> list[WorkflowArtifact]:
    """Deduplicate artifacts while preserving first-seen order."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[WorkflowArtifact] = []
    for artifact in artifacts:
        producer = artifact.producer_step_id or ""
        # Include producer in the key so two steps can emit the same path intentionally.
        key = (artifact.path, artifact.mime, producer)
        if key in seen:
            continue
        seen.add(key)
        unique.append(artifact)
    return unique


def collect_artifacts(
    *,
    step_results: Mapping[str, WorkflowStepResult],
    execution_order: Sequence[str],
) -> list[WorkflowArtifact]:
    """Collect ordered unique artifacts across all step results."""
    artifacts: list[WorkflowArtifact] = []
    for step_id in execution_order:
        step_result = step_results.get(step_id)
        if step_result is None:
            continue
        artifacts.extend(step_result.artifacts)
    return dedupe_artifacts(artifacts)


def resolve_final_output(
    *,
    step_results: Mapping[str, WorkflowStepResult],
    execution_order: Sequence[str],
) -> object:
    """Select one canonical final output payload from step results."""
    # Walk backward so the latest successful step wins, mirroring user-facing execution flow.
    for step_id in reversed(execution_order):
        step_result = step_results.get(step_id)
        if step_result is None:
            continue
        if step_result.status == "completed" and step_result.success:
            if "final_output" in step_result.output:
                return step_result.output["final_output"]
            return dict(step_result.output)
    return {}


__all__ = ["collect_artifacts", "dedupe_artifacts", "resolve_final_output"]
