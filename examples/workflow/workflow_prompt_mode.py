r"""# Workflow / Workflow Prompt Mode.

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
    C --> D["WorkflowRuntime schedules step graph (AgentStep, LogicStep, ToolStep)"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "agent_branch_run": {
       "error": null,
       "example": "workflow/workflow_prompt_mode.py",
       "execution_order": [
         "router",
         "draft_agent",
         "parse_agent_json",
         "finalize_agent",
         "draft_template",
         "finalize_template"
       ],
       "final_output": {
         "branch": "agent",
         "summary": "Use one runtime that fuses core, script, and MCP tools.",
         "title": "Deterministic workflow memo"
       },
       "success": true,
       "terminated_reason": null,
       "trace": {
         "request_id": "example-workflow-prompt-design-agent-001",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-workflow-prompt-design-agent-001.jsonl"
       }
     },
     "template_branch_run": {
       "error": null,
       "example": "workflow/workflow_prompt_mode.py",
       "execution_order": [
         "router",
         "draft_agent",
         "parse_agent_json",
         "finalize_agent",
         "draft_template",
         "finalize_template"
       ],
       "final_output": {
         "branch": "template",
         "summary": "Template mode output for: template: Produce a deterministic fallback brief for manufacturabili...
         "title": "Template fallback design brief"
       },
       "success": true,
       "terminated_reason": null,
       "trace": {
         "request_id": "example-workflow-prompt-design-template-001",
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
    AgentStep,
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
    final_output = result.final_output
    if isinstance(final_output, dict):
        compact_final_output = {
            "branch": final_output.get("branch"),
            "title": final_output.get("title"),
            "summary": final_output.get("summary"),
        }
    else:
        compact_final_output = final_output
    return result.summary(
        details={
            "execution_order": list(result.execution_order),
            "final_output_compact": compact_final_output,
        },
    )


def main() -> None:
    """Run reusable prompt-mode workflow for two routed design requests."""
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    writer_agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)

    workflow_steps = [
        LogicStep(
            step_id="router",
            handler=lambda context: {
                "route": ("template_path" if str(context["prompt"]).lower().startswith("template:") else "agent_path")
            },
            route_map={
                "agent_path": ("draft_agent",),
                "template_path": ("draft_template",),
            },
        ),
        AgentStep(
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
                "title": context["dependency_results"]["parse_agent_json"]["output"]["result"]["json"].get("title", ""),
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

    agent_request_id = "example-workflow-prompt-design-agent-001"
    template_request_id = "example-workflow-prompt-design-template-001"
    try:
        agent_result = workflow.run(
            "Draft a design brief for reducing onboarding friction in a medical-device setup flow.",
            request_id=agent_request_id,
        )
        template_result = workflow.run(
            ("template: Produce a deterministic fallback brief for manufacturability review findings."),
            request_id=template_request_id,
        )
    finally:
        llm_client.close()

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
