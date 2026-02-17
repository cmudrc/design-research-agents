"""Runnable example for ``TreeSearchPattern`` reasoning."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.workflow import TreeSearchPattern


def _generator(context: Mapping[str, object]) -> list[dict[str, object]]:
    """Generate deterministic child candidates for one search depth.

    Args:
        context: Delegate input containing task/depth metadata.

    Returns:
        Candidate payloads for the requested depth.
    """
    depth = int(context.get("depth", 0))
    if depth == 1:
        return [
            {"concept": "lightweight frame", "score_hint": 0.45},
            {"concept": "modular frame", "score_hint": 0.7},
        ]
    return [
        {"concept": "modular frame + fail-safe", "score_hint": 0.92},
        {"concept": "modular frame + low cost", "score_hint": 0.61},
    ]


def _evaluator(context: Mapping[str, object]) -> float:
    """Return a deterministic score for a candidate payload.

    Args:
        context: Delegate input containing one candidate.

    Returns:
        Candidate score hint converted to float.
    """
    candidate = context.get("candidate")
    if isinstance(candidate, Mapping):
        score = candidate.get("score_hint")
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0


def main() -> None:
    """Run one tree-search reasoning workflow and print result."""
    pattern = TreeSearchPattern(
        generator_delegate=_generator,
        evaluator_delegate=_evaluator,
        max_depth=2,
        branch_factor=2,
        beam_width=1,
    )
    result = pattern.run("Find a robust concept architecture.")
    print(result.asdict())


if __name__ == "__main__":
    main()
