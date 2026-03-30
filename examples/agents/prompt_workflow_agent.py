"""# Agents / Prompt Workflow Agent.

## Introduction
This example shows how to package a prompt-mode ``Workflow`` as a reusable study agent for deterministic
design experiments. ``PromptWorkflowAgent`` keeps the workflow itself simple while moving packaged-problem,
run-spec, and condition formatting into one explicit prompt builder.

## Technical Implementation
1. Define tiny local study packet dataclasses so the example stays dependency-light and deterministic.
2. Build a prompt-mode ``Workflow`` with logic steps that capture the resolved study prompt and emit one final
   summary payload.
3. Wrap that workflow in ``PromptWorkflowAgent`` with a prompt builder that converts study metadata into one
   canonical prompt string.
4. Run the delegate with a fixed ``request_id`` and print a compact JSON payload for docs and regression tests.

```mermaid
flowchart LR
    A["Problem packet + run spec + condition"] --> B["PromptWorkflowAgent(prompt_builder)"]
    B --> C["Prompt-mode Workflow"]
    C --> D["capture_study_prompt"]
    D --> E["emit_summary"]
    E --> F["JSON payload"]
```

## Expected Results

Example output shape:

.. code-block:: text

   {
     "workflow_mermaid": "flowchart LR ...",
     "summary": {
       "success": true,
       "final_output": {
        "request_id": "example-prompt-workflow-agent-001",
         "study_prompt": "Problem: cooling_plate_redesign...",
         "workflow_step": "emit_summary"
       },
       "terminated_reason": null,
       "error": null,
       "trace": {
         "request_id": "example-workflow-study-delegate-001"
       }
     }
   }

## References
- `HELM: Holistic Evaluation of Language Models <https://arxiv.org/abs/2211.09110>`_
- `Python dataclasses <https://docs.python.org/3/library/dataclasses.html>`_
- `Design Research Agents documentation <https://cmudrc.github.io/design-research-agents/>`_
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import design_research_agents as drag

WORKFLOW_DIAGRAM_DIRECTION = "LR"


@dataclass(frozen=True)
class _ProblemPacket:
    problem_id: str
    brief: str


@dataclass(frozen=True)
class _RunSpec:
    run_id: str
    objective: str


@dataclass(frozen=True)
class _Condition:
    condition_id: str
    budget_label: str


def _capture_study_prompt(context: dict[str, Any]) -> dict[str, object]:
    """Capture the resolved study prompt and request id inside the workflow."""
    workflow_metadata = context.get("_workflow")
    request_id = ""
    if isinstance(workflow_metadata, dict):
        raw_request_id = workflow_metadata.get("request_id")
        if isinstance(raw_request_id, str):
            request_id = raw_request_id
    return {
        "study_prompt": str(context["prompt"]),
        "request_id": request_id,
    }


def _emit_summary(context: dict[str, Any]) -> dict[str, object]:
    """Emit the canonical final_output payload for this example run."""
    capture_output = context["dependency_results"]["capture_study_prompt"]["output"]
    return {
        "final_output": {
            "request_id": capture_output["request_id"],
            "study_prompt": capture_output["study_prompt"],
            "workflow_step": str(context["_workflow"]["step_id"]),
        }
    }


def build_example_workflow() -> drag.Workflow:
    """Build the prompt-mode workflow wrapped by the prompt workflow agent."""
    return drag.Workflow(
        steps=[
            drag.LogicStep(
                step_id="capture_study_prompt",
                handler=_capture_study_prompt,
            ),
            drag.LogicStep(
                step_id="emit_summary",
                dependencies=("capture_study_prompt",),
                handler=_emit_summary,
            ),
        ]
    )


def _build_study_prompt(problem_packet: object, run_spec: object, condition: object) -> str:
    """Translate study packet objects into one deterministic workflow prompt."""
    if not isinstance(problem_packet, _ProblemPacket):
        raise TypeError("Expected _ProblemPacket for problem_packet.")
    if not isinstance(run_spec, _RunSpec):
        raise TypeError("Expected _RunSpec for run_spec.")
    if not isinstance(condition, _Condition):
        raise TypeError("Expected _Condition for condition.")
    return (
        f"Problem: {problem_packet.problem_id}. Brief: {problem_packet.brief} "
        f"Run: {run_spec.run_id}. Objective: {run_spec.objective}. "
        f"Condition: {condition.condition_id} ({condition.budget_label})."
    )


def main() -> None:
    """Run the workflow-backed study agent and print a deterministic summary payload."""
    agent = drag.PromptWorkflowAgent(
        workflow=build_example_workflow(),
        prompt_builder=_build_study_prompt,
    )
    problem_packet = _ProblemPacket(
        problem_id="cooling_plate_redesign",
        brief="Reduce pressure drop while preserving manufacturability.",
    )
    run_spec = _RunSpec(
        run_id="study-run-12",
        objective="Summarize the design-study setup for a control workflow.",
    )
    condition = _Condition(
        condition_id="control_workflow",
        budget_label="single-pass",
    )
    result = agent.run(
        "Fallback prompts are also supported, but this example uses study dependencies.",
        request_id="example-prompt-workflow-agent-001",
        dependencies={
            "problem_packet": problem_packet,
            "run_spec": run_spec,
            "condition": condition,
        },
    )
    payload = {
        "workflow_mermaid": agent.workflow.to_mermaid(direction=WORKFLOW_DIAGRAM_DIRECTION),
        "summary": result.summary(),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
