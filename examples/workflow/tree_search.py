"""Run traced ``TreeSearchPattern`` for design concept selection.

Expected observations:
- search evaluates deterministic candidates across depths.
- ``final_output`` identifies best-scoring concept branch.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from design_research_agents import Tracer, TreeSearchPattern


def _generator(context: Mapping[str, object]) -> list[dict[str, object]]:
    """Generate deterministic design candidates for one search depth."""
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
    """Return deterministic score for one candidate payload."""
    candidate = context.get("candidate")
    if isinstance(candidate, Mapping):
        score = candidate.get("score_hint")
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0


def main() -> None:
    """Run one tree-search workflow and print JSON summary."""
    request_id = "example-workflow-tree-search-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    pattern = TreeSearchPattern(
        generator_delegate=_generator,
        evaluator_delegate=_evaluator,
        max_depth=2,
        branch_factor=2,
        beam_width=1,
        tracer=tracer,
    )
    result = pattern.run(
        "Find the most robust concept architecture for a serviceable edge-device enclosure.",
        request_id=request_id,
    )
    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/tree_search.py",
        "success": result.success,
        "final_output": result.final_output,
        "best_candidate": output.get("best_candidate"),
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
