"""# Workflow / Workflow Prompt Mode.

## Introduction
ReAct and Plan-and-Solve motivate explicit control over reasoning phases, and JSON Schema formalizes
structured inputs/outputs when prompt-mode steps need predictable contracts. This example shows prompt-mode
workflow composition with agent, logic, and tool steps under one runtime, including one packaged-problem-like
object passed directly to ``Workflow.run(...)``.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``,
   once from a packaged-problem-like object and once from a plain fallback string prompt.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "agent_branch_run": {
       "success": true,
       "final_output": "<evaluation-ready final_output payload>",
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
from collections.abc import Mapping
from pathlib import Path

import design_research_agents as drag

WORKFLOW_DIAGRAM_DIRECTION = "LR"


class _DocDelegate:
    """Minimal delegate stub used only for docs-diagram workflow construction."""

    def run(self, prompt: str, *, request_id: str | None = None, dependencies: object | None = None) -> object:
        del prompt, request_id, dependencies
        raise RuntimeError("Docs-only delegate stub should not be executed.")


class _ExampleProblemMetadata:
    def __init__(self, *, problem_id: str, title: str, kind: str) -> None:
        self.problem_id = problem_id
        self.title = title
        self.kind = kind


class _ExamplePackagedProblem:
    """Tiny packaged-problem stand-in used to document prompt-like workflow inputs."""

    def __init__(self) -> None:
        self.metadata = _ExampleProblemMetadata(
            problem_id="workflow-prompt-problem-001",
            title="Reduce onboarding friction",
            kind="design-brief",
        )
        self.candidate_kind = "json-brief"
        self.family = "workflow-example"

    def render_brief(self) -> str:
        return "Draft a design brief for reducing onboarding friction in a medical-device setup flow."


def _summarize_run(result: drag.ExecutionResult) -> dict[str, object]:
    return result.summary()


def _problem_metadata_from_context(context: Mapping[str, object]) -> dict[str, object]:
    metadata = context.get("problem_metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def build_example_workflow(
    *,
    tracer: drag.Tracer | None = None,
    tool_runtime: object | None = None,
    writer_agent: object | None = None,
) -> drag.Workflow:
    """Build the routed prompt-mode workflow used for docs diagrams and runtime execution."""
    resolved_writer_agent = writer_agent or _DocDelegate()
    return drag.Workflow(
        tool_runtime=tool_runtime,
        tracer=tracer,
        steps=[
            drag.LogicStep(
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
            drag.DelegateStep(
                step_id="draft_agent",
                delegate=resolved_writer_agent,
                dependencies=("router",),
                prompt_builder=lambda context: (
                    f"Write one JSON object with keys title and summary for this design request: {context['prompt']}"
                ),
            ),
            drag.ToolStep(
                step_id="parse_agent_json",
                tool_name="text.extract_json",
                dependencies=("draft_agent",),
                input_builder=lambda context: {
                    "text": context["dependency_results"]["draft_agent"]["output"]["output"]["model_text"]
                },
            ),
            drag.LogicStep(
                step_id="finalize_agent",
                dependencies=("parse_agent_json",),
                handler=lambda context: {
                    "branch": "agent",
                    "problem_id": _problem_metadata_from_context(context).get("problem_id", ""),
                    "candidate_kind": _problem_metadata_from_context(context).get("candidate_kind", ""),
                    "title": context["dependency_results"]["parse_agent_json"]["output"]["result"]["json"].get(
                        "title", ""
                    ),
                    "summary": context["dependency_results"]["parse_agent_json"]["output"]["result"]["json"].get(
                        "summary", ""
                    ),
                },
            ),
            drag.LogicStep(
                step_id="draft_template",
                dependencies=("router",),
                handler=lambda context: {
                    "title": "Template fallback design brief",
                    "summary": f"Template mode output for: {context['prompt']}",
                },
            ),
            drag.LogicStep(
                step_id="finalize_template",
                dependencies=("draft_template",),
                handler=lambda context: {
                    "branch": "template",
                    "title": context["dependency_results"]["draft_template"]["output"]["title"],
                    "summary": context["dependency_results"]["draft_template"]["output"]["summary"],
                },
            ),
        ],
    )


def main() -> None:
    """Run reusable prompt-mode workflow for one packaged problem and one fallback prompt."""
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    # Run the prompt-mode workflow using public runtime surfaces. Using this with statement will automatically
    # shut down the managed client and tool runtime when the example is done.
    with drag.Toolbox() as tool_runtime, drag.LlamaCppServerLLMClient() as llm_client:
        writer_agent = drag.DirectLLMCall(llm_client=llm_client, tracer=tracer)
        workflow = build_example_workflow(
            tracer=tracer,
            tool_runtime=tool_runtime,
            writer_agent=writer_agent,
        )

        # Keep per-branch request ids stable so prompt-mode variants are easy to compare.
        agent_request_id = "example-workflow-prompt-design-agent-001"
        # Keep per-branch request ids stable so prompt-mode variants are easy to compare.
        template_request_id = "example-workflow-prompt-design-template-001"
        packaged_problem = _ExamplePackagedProblem()
        agent_result = workflow.run(
            packaged_problem,
            request_id=agent_request_id,
        )
        template_result = workflow.run(
            ("template: Produce a deterministic fallback brief for manufacturability review findings."),
            request_id=template_request_id,
        )

    # Print the results
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
