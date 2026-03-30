"""# Agents / Seeded Random Baseline Agent.

## Introduction
This example treats a seeded random baseline as a first-class study participant for packaged-problem-style
experiments. It keeps the setup dependency-light by using a local decision-problem stub that mirrors the
public candidate-iteration contract instead of importing sibling repositories, while still using the same
``run(prompt, dependencies=...)`` contract as the other public agents.

## Technical Implementation
1. Define a small packaged-problem-style decision stub with ``iter_candidates()`` and ``evaluate()``.
2. Run ``SeededRandomBaselineAgent`` with a fixed seed and pass the packaged problem through
   ``dependencies`` so the sampled control candidate is reproducible.
3. Compare the random control condition against a deterministic greedy baseline that always picks the
   highest-scoring candidate.
4. Print JSON that could be dropped into lightweight experiment wiring or docs examples.

```mermaid
flowchart LR
    A["Local decision problem stub"] --> B["SeededRandomBaselineAgent(seed=7)"]
    A --> C["Greedy comparator"]
    B --> D["Random control candidate"]
    C --> E["Deterministic candidate"]
    D --> F["JSON comparison output"]
    E --> F
```

## Expected Results

Example output shape:

.. code-block:: text

   {
     "problem_id": "local_heat_sink_layout",
     "random_condition": {
       "candidate": {"fin_count": 6.0, "gap_mm": 2.0, "wall_mm": 1.0},
       "score": 0.89
     },
     "greedy_condition": {
       "candidate": {"fin_count": 8.0, "gap_mm": 2.0, "wall_mm": 1.5},
       "score": 1.065
     }
   }

## References
- `Python random module <https://docs.python.org/3/library/random.html>`_
- `HELM: Holistic Evaluation of Language Models <https://arxiv.org/abs/2211.09110>`_
- `Design Research Agents documentation <https://cmudrc.github.io/design-research-agents/>`_
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import design_research_agents as drag


@dataclass(frozen=True)
class _ProblemMetadata:
    """Minimal metadata stub for the local example problem."""

    problem_id: str


class _HeatSinkDecisionProblem:
    """Tiny local decision problem exposing the public candidate iterator."""

    def __init__(self) -> None:
        """Initialize the local benchmark stub."""
        self.metadata = _ProblemMetadata(problem_id="local_heat_sink_layout")
        self._candidates = (
            {"fin_count": 4.0, "gap_mm": 2.5, "wall_mm": 1.0},
            {"fin_count": 6.0, "gap_mm": 2.0, "wall_mm": 1.0},
            {"fin_count": 8.0, "gap_mm": 2.0, "wall_mm": 1.5},
        )

    def iter_candidates(self) -> tuple[dict[str, float], ...]:
        """Return admissible candidates in deterministic order."""
        return tuple(dict(candidate) for candidate in self._candidates)

    def evaluate(self, candidate: dict[str, float]) -> dict[str, float]:
        """Return a simple scalar score for one local candidate."""
        score = 0.05 * candidate["fin_count"] + 0.22 * candidate["gap_mm"] + 0.15 * candidate["wall_mm"]
        return {"objective_value": round(score, 4)}


def _greedy_candidate(problem: _HeatSinkDecisionProblem) -> dict[str, float]:
    """Return the highest-scoring deterministic candidate."""
    candidates = problem.iter_candidates()
    return max(candidates, key=lambda candidate: problem.evaluate(candidate)["objective_value"])


def main() -> None:
    """Run the seeded random baseline next to a deterministic comparator."""
    problem = _HeatSinkDecisionProblem()
    agent = drag.SeededRandomBaselineAgent(seed=7)
    random_run = agent.run(
        "Sample one seeded random control candidate for the local heat sink study.",
        dependencies={"problem": problem},
    )
    random_candidate = random_run.output_dict("final_output")

    greedy_candidate = _greedy_candidate(problem)
    payload = {
        "problem_id": problem.metadata.problem_id,
        "random_condition": {
            "candidate": random_candidate,
            "score": problem.evaluate(random_candidate)["objective_value"],
            "metadata": random_run.metadata,
        },
        "greedy_condition": {
            "candidate": greedy_candidate,
            "score": problem.evaluate(greedy_candidate)["objective_value"],
        },
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
