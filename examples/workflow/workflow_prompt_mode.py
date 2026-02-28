"""# Workflow / Workflow Prompt Mode.

## Introduction
ReAct and Plan-and-Solve motivate explicit control over reasoning phases, and JSON Schema formalizes
structured inputs/outputs when prompt-mode steps need predictable contracts. This example shows prompt-mode
workflow composition with agent, logic, and tool steps under one runtime.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["Workflow.run(...)"]
    C --> D["WorkflowRuntime schedules step graph (DelegateStep, LogicStep, ToolStep)"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "agent_branch_run": {
       "success": true,
       "final_output": "<example-specific payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     },
     "template_branch_run": {
       "success": true,
       "final_output": "<example-specific payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     }
   }

## References
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DelegateStep,
    DirectLLMCall,
    ExecutionResult,
    LlamaCppServerLLMClient,
    LogicStep,
    Toolbox,
    ToolStep,
    Tracer,
    Workflow,
)


def _summarize_run(result: ExecutionResult) -> dict[str, object]:
    return result.summary()


def main() -> None:
    """Run reusable prompt-mode workflow for two routed design requests."""
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    with LlamaCppServerLLMClient() as llm_client:
        tool_runtime = Toolbox()
        writer_agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)

        workflow_steps = [
            LogicStep(
                step_id="router",
                handler=lambda context: {
                    "route": (
                        "template_path" if str(context["prompt"]).lower().startswith("template:") else "agent_path"
                    )
                },
                route_map={
                    "agent_path": ("draft_agent",),
                    "template_path": ("draft_template",),
                },
            ),
            DelegateStep(
                step_id="draft_agent",
                delegate=writer_agent,
                dependencies=("router",),
                prompt_builder=lambda context: (
                    f"Write one JSON object with keys title and summary for this design request: {context['prompt']}"
                ),
            ),
            ToolStep(
                step_id="parse_agent_json",
                tool_name="text.extract_json",
                dependencies=("draft_agent",),
                input_builder=lambda context: {
                    "text": context["dependency_results"]["draft_agent"]["output"]["output"]["model_text"]
                },
            ),
            LogicStep(
                step_id="finalize_agent",
                dependencies=("parse_agent_json",),
                handler=lambda context: {
                    "branch": "agent",
                    "title": context["dependency_results"]["parse_agent_json"]["output"]["result"]["json"].get(
                        "title", ""
                    ),
                    "summary": context["dependency_results"]["parse_agent_json"]["output"]["result"]["json"].get(
                        "summary", ""
                    ),
                },
            ),
            LogicStep(
                step_id="draft_template",
                dependencies=("router",),
                handler=lambda context: {
                    "title": "Template fallback design brief",
                    "summary": f"Template mode output for: {context['prompt']}",
                },
            ),
            LogicStep(
                step_id="finalize_template",
                dependencies=("draft_template",),
                handler=lambda context: {
                    "branch": "template",
                    "title": context["dependency_results"]["draft_template"]["output"]["title"],
                    "summary": context["dependency_results"]["draft_template"]["output"]["summary"],
                },
            ),
        ]

        workflow = Workflow(
            tool_runtime=tool_runtime,
            steps=workflow_steps,
            tracer=tracer,
        )

        # Keep per-branch request ids stable so prompt-mode variants are easy to compare.
        agent_request_id = "example-workflow-prompt-design-agent-001"
        # Keep per-branch request ids stable so prompt-mode variants are easy to compare.
        template_request_id = "example-workflow-prompt-design-template-001"
        agent_result = workflow.run(
            "Draft a design brief for reducing onboarding friction in a medical-device setup flow.",
            request_id=agent_request_id,
        )
        template_result = workflow.run(
            ("template: Produce a deterministic fallback brief for manufacturability review findings."),
            request_id=template_request_id,
        )

    print(
        json.dumps(
            {
                "agent_branch_run": _summarize_run(agent_result),
                "template_branch_run": _summarize_run(template_result),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
