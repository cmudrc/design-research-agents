"""Artifact normalization tests for canonical pattern results."""

from __future__ import annotations

import pytest

from design_research_agents._runtime._patterns import build_pattern_output


@pytest.mark.parametrize(
    ("artifacts", "expected"),
    [
        (("trace.json", "metrics.json"), ["trace.json", "metrics.json"]),
        (range(2), [0, 1]),
        ("trace.json", []),
    ],
)
def test_pattern_output_normalizes_artifact_sequences(
    artifacts: object,
    expected: list[object],
) -> None:
    output = build_pattern_output(
        final_output={},
        terminated_reason="complete",
        details={},
        workflow_payload={},
        artifacts=artifacts,
    )

    assert output["artifacts"] == expected
