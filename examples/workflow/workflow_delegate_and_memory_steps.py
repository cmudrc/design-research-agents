"""# Workflow / Workflow Delegate And Memory Steps.

## Introduction
Generative Agents and MemGPT both emphasize durable memory as a first-class runtime primitive, while AutoGen
demonstrates delegation across specialized roles. This example composes delegate and memory steps in a
single workflow so context propagation and role handoff remain explicit.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Persist and query context via ``SQLiteMemoryStore`` to demonstrate memory-backed workflow behavior.
5. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
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

## References
- `Generative Agents <https://arxiv.org/abs/2304.03442>`_
- `MemGPT <https://arxiv.org/abs/2310.08560>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
"""

from __future__ import annotations

import json
from pathlib import Path

import design_research_agents as drag
from design_research_agents.memory import SQLiteMemoryStore

WORKFLOW_DIAGRAM_DIRECTION = "LR"


class _DocDelegate:
    """Minimal delegate stub used only for docs-diagram workflow construction."""

    def run(self, prompt: str, *, request_id: str | None = None, dependencies: object | None = None) -> object:
        del prompt, request_id, dependencies
        raise RuntimeError("Docs-only delegate stub should not be executed.")


def build_example_workflow(
    *,
    tracer: drag.Tracer | None = None,
    memory_store: object | None = None,
    manufacturing_peer: object | None = None,
    reliability_peer: object | None = None,
) -> drag.Workflow:
    """Build the delegate-memory workflow used for docs diagrams and runtime execution."""
    resolved_manufacturing_peer = manufacturing_peer or _DocDelegate()
    resolved_reliability_peer = reliability_peer or _DocDelegate()
    return drag.Workflow(
        tool_runtime=None,
        memory_store=memory_store,
        tracer=tracer,
        input_schema={"type": "object"},
        steps=[
            drag.MemoryWriteStep(
                step_id="seed_constraints",
                namespace="design_constraints",
                records_builder=lambda _context: [
                    {
                        "content": "Constraint: reduce service time by at least 20 percent.",
                        "metadata": {"kind": "constraint"},
                    },
                    {
                        "content": "Constraint: preserve ingress protection sealing.",
                        "metadata": {"kind": "constraint"},
                    },
                ],
            ),
            drag.MemoryReadStep(
                step_id="read_constraints",
                namespace="design_constraints",
                dependencies=("seed_constraints",),
                top_k=5,
                query_builder=lambda _context: {
                    "text": "service time constraint",
                    "metadata_filters": {"kind": "constraint"},
                },
            ),
            drag.DelegateBatchStep(
                step_id="peer_batch",
                dependencies=("read_constraints",),
                fail_fast=False,
                calls_builder=lambda context: [
                    {
                        "call_id": "manufacturing_peer",
                        "delegate": resolved_manufacturing_peer,
                        "prompt": (
                            "Propose manufacturing-friendly maintenance improvements using "
                            "retrieved constraints count="
                            f"{context['dependency_results']['read_constraints']['output']['count']}."
                        ),
                    },
                    {
                        "call_id": "reliability_peer",
                        "delegate": resolved_reliability_peer,
                        "prompt": "Propose reliability-focused maintenance improvements.",
                    },
                ],
            ),
            drag.LogicStep(
                step_id="finalize",
                dependencies=("read_constraints", "peer_batch"),
                handler=lambda context: {
                    "constraints_found": (context["dependency_results"]["read_constraints"]["output"]["count"]),
                    "delegate_calls": len(context["dependency_results"]["peer_batch"]["output"].get("results", [])),
                    "final_delegate_output": (
                        context["dependency_results"]["peer_batch"]["output"].get("final_output")
                    ),
                },
            ),
        ],
    )


def main() -> None:
    """Execute memory and delegate-batch primitives in one traced workflow."""
    # Stable request ids keep workflow trace artifacts deterministic for docs snapshots.
    request_id = "example-workflow-delegate-memory-design-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    db_path = Path("artifacts/examples/workflow_delegate_and_memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Reset persisted state so each example run starts from the same memory baseline.
    if db_path.exists():
        db_path.unlink()

    # Run the workflow using public runtime surfaces. Using this with statement will automatically close the
    # memory store and managed client when the example is done.
    with (
        SQLiteMemoryStore(db_path=db_path) as store,
        drag.LlamaCppServerLLMClient() as llm_client,
    ):
        # Two delegates share the same backend client to model role-specific prompts over one transport.
        manufacturing_peer = drag.DirectLLMCall(llm_client=llm_client, tracer=tracer)
        reliability_peer = drag.DirectLLMCall(llm_client=llm_client, tracer=tracer)

        workflow = build_example_workflow(
            tracer=tracer,
            memory_store=store,
            manufacturing_peer=manufacturing_peer,
            reliability_peer=reliability_peer,
        )
        result = workflow.run({}, request_id=request_id)

    # Print the results
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
