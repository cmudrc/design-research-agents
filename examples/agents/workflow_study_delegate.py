"""# Agents / Workflow Study Delegate.

## Introduction
This example shows how ``WorkflowStudyDelegate`` can wrap a prompt-mode workflow for
packaged-problem-style study execution without introducing external dependencies. The
delegate uses structured study metadata to build the workflow prompt while keeping the
runtime surface the same ``run(prompt, dependencies=...)`` contract used elsewhere in
the public API.

## Technical Implementation
1. Define tiny local study packet stubs for the problem, run, and condition inputs.
2. Build a prompt-mode ``Workflow`` that echoes the resolved study prompt and metadata.
3. Wrap that workflow with ``WorkflowStudyDelegate`` and a deterministic prompt builder.
4. Print a small JSON payload showing the compiled prompt and workflow output.

```mermaid
flowchart LR
    A["Problem packet"] --> D["WorkflowStudyDelegate"]
    B["Run spec"] --> D
    C["Condition"] --> D
    D --> E["Prompt-mode Workflow"]
    E --> F["JSON study output"]
```

## Expected Results

Example output shape:

.. code-block:: text

   {
     "compiled_input": "Study local heat sink concept alternatives under cond-baseline for run-3.",
     "final_output": {
       "condition_id": "cond-baseline",
       "problem_brief": "Compare local heat sink concept alternatives.",
       "prompt": "Study local heat sink concept alternatives under cond-baseline for run-3.",
       "run_id": "run-3"
     }
   }

## References
- `Workflow examples <https://cmudrc.github.io/design-research-agents/examples/workflow/index.html>`_
- `Pattern overview <https://cmudrc.github.io/design-research-agents/patterns/overview.html>`_
- `Design Research Agents documentation <https://cmudrc.github.io/design-research-agents/>`_
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import design_research_agents as drag


@dataclass(frozen=True)
class _ProblemPacket:
    """Minimal local problem-packet stub for the example."""

    brief: str


@dataclass(frozen=True)
class _RunSpec:
    """Minimal local run metadata stub."""

    run_id: str


@dataclass(frozen=True)
class _Condition:
    """Minimal local study-condition stub."""

    condition_id: str


def main() -> None:
    """Run one lightweight workflow study example and print the payload."""
    workflow = drag.Workflow(
        steps=(
            drag.LogicStep(
                step_id="emit_study_payload",
                handler=lambda context: {
                    "final_output": {
                        "prompt": context.input,
                        "problem_brief": context.dependencies["problem_packet"].brief,
                        "run_id": context.dependencies["run_spec"].run_id,
                        "condition_id": context.dependencies["condition"].condition_id,
                    }
                },
            ),
        )
    )
    delegate = drag.WorkflowStudyDelegate(
        workflow=workflow,
        prompt_builder=lambda problem_packet, run_spec, condition: (
            f"Study {problem_packet.brief.lower()} under {condition.condition_id} for {run_spec.run_id}."
        ),
    )

    dependencies = {
        "problem_packet": _ProblemPacket(brief="Compare local heat sink concept alternatives"),
        "run_spec": _RunSpec(run_id="run-3"),
        "condition": _Condition(condition_id="cond-baseline"),
    }
    compiled = delegate.compile("ignored fallback prompt", dependencies=dependencies)
    result = delegate.run("ignored fallback prompt", dependencies=dependencies)

    payload = {
        "compiled_input": compiled.input,
        "final_output": result.output_dict("final_output"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
