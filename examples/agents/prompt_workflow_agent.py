"""# Agents / Prompt Workflow Agent.

## Introduction
This example shows how to package a JSON prompt-mode ``Workflow`` as a reusable study agent for deterministic
design experiments. ``PromptWorkflowAgent`` keeps study prompt construction separate from the workflow that
turns model output into one structured JSON payload.

## Technical Implementation
1. Define tiny local study packet dataclasses plus a public ``StudyCondition``.
2. Build a prompt-mode ``Workflow`` with ``build_json_prompt_workflow(...)`` and a tiny deterministic LLM
   client.
3. Wrap that workflow in ``PromptWorkflowAgent`` with a prompt builder that converts study metadata into one
   canonical prompt string.
4. Execute an ``AgentRunRequest`` through ``execute_agent_request(...)`` and print a compact JSON payload for
   docs and regression tests.

```mermaid
flowchart LR
    A["Problem packet + run spec + condition"] --> B["PromptWorkflowAgent(prompt_builder)"]
    B --> C["build_json_prompt_workflow"]
    C --> D["json_response"]
    D --> E["JSON payload"]
```

## Expected Results

Example output shape:

.. code-block:: text

   {
     "workflow_mermaid": "flowchart LR ...",
     "execution": {
       "output": {
         "study_prompt": "Problem: cooling_plate_redesign...",
         "workflow_step": "json_response"
       },
       "metrics": {},
       "event_count": 1
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


class _DeterministicJSONClient:
    """Small deterministic LLM client used to keep the example offline."""

    def generate(self, request: drag.LLMRequest) -> drag.LLMResponse:
        """Return a valid JSON response echoing the study prompt."""
        prompt = request.messages[-1].content
        return drag.LLMResponse(
            text=json.dumps(
                {
                    "study_prompt": prompt,
                    "workflow_step": "json_response",
                    "prompt_character_count": len(prompt),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            model="deterministic-json-client",
            provider="example",
            usage={"prompt_tokens": 42, "completion_tokens": 24},
        )


def build_example_workflow() -> drag.Workflow:
    """Build the prompt-mode workflow wrapped by the prompt workflow agent."""
    return drag.build_json_prompt_workflow(
        llm_client=_DeterministicJSONClient(),
        response_schema={
            "type": "object",
            "properties": {
                "study_prompt": {"type": "string"},
                "workflow_step": {"type": "string"},
                "prompt_character_count": {"type": "number"},
            },
            "required": ["study_prompt", "workflow_step", "prompt_character_count"],
        },
        request_metadata={"example": "prompt_workflow_agent"},
        default_request_id_prefix="example-prompt-workflow-agent",
    )


def _build_study_prompt(problem_packet: object, run_spec: object, condition: object) -> str:
    """Translate study packet objects into one deterministic workflow prompt."""
    if not isinstance(problem_packet, _ProblemPacket):
        raise TypeError("Expected _ProblemPacket for problem_packet.")
    if not isinstance(run_spec, _RunSpec):
        raise TypeError("Expected _RunSpec for run_spec.")
    if not isinstance(condition, drag.StudyCondition):
        raise TypeError("Expected StudyCondition for condition.")
    budget_label = str(condition.metadata.get("budget_label", "unspecified"))
    return (
        f"Problem: {problem_packet.problem_id}. Brief: {problem_packet.brief} "
        f"Run: {run_spec.run_id}. Objective: {run_spec.objective} "
        f"Condition: {condition.condition_id} ({budget_label})."
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
    condition = drag.StudyCondition(
        condition_id="control_workflow",
        label="Control workflow",
        metadata={"budget_label": "single-pass"},
    )
    run_request = drag.AgentRunRequest(
        agent_ref=agent,
        prompt="Fallback prompts are also supported, but this example uses study dependencies.",
        request_id="example-prompt-workflow-agent-001",
        dependencies={
            "problem_packet": problem_packet,
            "run_spec": run_spec,
            "condition": condition,
        },
    )
    execution: drag.AgentExecutionEnvelope = drag.execute_agent_request(run_request)
    compatibility_execution = drag.execute_agent_run(
        lambda prompt: {"output": {"text": prompt}, "metadata": {"path": "execute_agent_run"}},
        prompt="Compatibility execution path.",
        request_id="example-prompt-workflow-agent-compat",
        dependencies={},
    )
    normalized_preview = drag.normalize_agent_execution(
        {"text": "Normalized execution preview."},
        request_id="example-prompt-workflow-agent-normalize",
    )
    payload = {
        "workflow_mermaid": agent.workflow.to_mermaid(direction=WORKFLOW_DIAGRAM_DIRECTION),
        "execution": {
            "output": execution.output,
            "metrics": execution.metrics,
            "trace_refs": execution.trace_refs,
            "metadata": execution.metadata,
            "event_count": len(execution.events),
        },
        "compatibility": {
            "output": compatibility_execution.output,
            "normalized_preview": normalized_preview.output,
        },
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
