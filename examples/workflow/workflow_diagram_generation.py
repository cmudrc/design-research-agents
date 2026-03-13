"""# Workflow / Workflow Diagram Generation.

## Introduction
Workflow definitions stay readable while they are small, but nested loops and conditional routing
get harder to inspect once orchestration grows. This example shows how to generate a deterministic
Mermaid diagram directly from a configured ``Workflow`` so the same topology can be reused in local
development, docs, and review discussions.


## Technical Implementation
1. Build a representative workflow using only the public workflow primitives.
2. Call ``Workflow.to_mermaid(direction="LR")`` to render the declared topology.
3. Call ``Workflow.to_svg(direction="LR")`` to emit a static SVG for notebooks, docs assets, or reviews.
4. Persist both diagram formats under ``artifacts/examples/`` for local inspection or docs reuse.
5. Print a compact JSON payload so example automation can verify the generated diagram shape.


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "diagram_path": "artifacts/examples/workflow_diagram.mmd",
     "svg_path": "artifacts/examples/workflow_diagram.svg",
     "line_count": 18,
     "starts_with": "flowchart LR",
     "svg_starts_with": "<svg",
     "contains_loop": true,
     "contains_route": true
   }

## References
- `Mermaid flowcharts <https://mermaid.js.org/syntax/flowchart.html>`_
- `Mermaid subgraphs <https://mermaid.js.org/syntax/flowchart.html#subgraphs>`_
- `Mermaid node shapes and labels <https://mermaid.js.org/syntax/flowchart.html#node-shapes>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag

WORKFLOW_DIAGRAM_DIRECTION = "LR"


def build_example_workflow() -> drag.Workflow:
    """Build a representative workflow with routing and loop structure."""
    return drag.Workflow(
        steps=[
            drag.LogicStep(
                step_id="prepare",
                handler=lambda _context: {"prompt_ready": True},
            ),
            drag.LoopStep(
                step_id="review_loop",
                dependencies=("prepare",),
                max_iterations=2,
                steps=(
                    drag.LogicStep(
                        step_id="router",
                        handler=lambda _context: {"route": "draft_path"},
                        route_map={
                            "draft_path": ("draft",),
                            "score_path": ("score",),
                        },
                    ),
                    drag.LogicStep(
                        step_id="draft",
                        dependencies=("router",),
                        handler=lambda _context: {"draft": "candidate"},
                    ),
                    drag.LogicStep(
                        step_id="score",
                        dependencies=("draft",),
                        handler=lambda _context: {"score": 0.9},
                    ),
                ),
            ),
            drag.LogicStep(
                step_id="publish",
                dependencies=("review_loop",),
                handler=lambda _context: {"published": True},
            ),
        ]
    )


def main() -> None:
    """Generate Mermaid and SVG output for a representative workflow."""
    workflow = build_example_workflow()
    diagram = workflow.to_mermaid(direction=WORKFLOW_DIAGRAM_DIRECTION)
    svg = workflow.to_svg(direction=WORKFLOW_DIAGRAM_DIRECTION)

    mermaid_path = Path("artifacts/examples/workflow_diagram.mmd")
    svg_path = Path("artifacts/examples/workflow_diagram.svg")
    mermaid_path.parent.mkdir(parents=True, exist_ok=True)
    mermaid_path.write_text(diagram + "\n", encoding="utf-8")
    svg_path.write_text(svg + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "diagram_path": mermaid_path.as_posix(),
                "svg_path": svg_path.as_posix(),
                "line_count": len(diagram.splitlines()),
                "starts_with": diagram.splitlines()[0],
                "svg_starts_with": svg.lstrip()[:4],
                "contains_loop": "Loop Body: review_loop" in diagram,
                "contains_route": "route=draft_path" in diagram,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
