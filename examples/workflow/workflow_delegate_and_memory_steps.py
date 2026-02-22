"""Run traced workflow covering delegate batch and memory read/write step APIs.

Expected observations:
- ``MemoryWriteStep`` seeds design constraints.
- ``MemoryReadStep`` retrieves relevant context records.
- ``DelegateBatchStep`` runs multiple deterministic delegates.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

from _support_deterministic import FixedDesignPeerAgent

from design_research_agents import (
    DelegateBatchStep,
    LogicStep,
    MemoryReadStep,
    MemoryWriteStep,
    Tracer,
    Workflow,
)
from design_research_agents.memory import SQLiteMemoryStore


def main() -> None:
    """Execute memory and delegate-batch primitives in one traced workflow."""
    request_id = "example-workflow-delegate-memory-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    db_path = Path("artifacts/examples/workflow_delegate_and_memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    store = SQLiteMemoryStore(db_path=db_path)

    workflow = Workflow(
        tool_runtime=None,
        memory_store=store,
        tracer=tracer,
        input_schema={"type": "object"},
        steps=[
            MemoryWriteStep(
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
            MemoryReadStep(
                step_id="read_constraints",
                namespace="design_constraints",
                dependencies=("seed_constraints",),
                top_k=5,
                query_builder=lambda _context: {
                    "text": "service time constraint",
                    "metadata_filters": {"kind": "constraint"},
                },
            ),
            DelegateBatchStep(
                step_id="peer_batch",
                dependencies=("read_constraints",),
                fail_fast=False,
                calls_builder=lambda context: [
                    {
                        "call_id": "manufacturing_peer",
                        "delegate": FixedDesignPeerAgent(
                            messages=["Use captive screws with standardized head type for faster maintenance."]
                        ),
                        "prompt": (
                            "Propose manufacturing-friendly maintenance improvements using "
                            "retrieved constraints count="
                            f"{context['dependency_results']['read_constraints']['output']['count']}."
                        ),
                    },
                    {
                        "call_id": "reliability_peer",
                        "delegate": FixedDesignPeerAgent(
                            messages=["Add gasket alignment features to preserve ingress protection after service."]
                        ),
                        "prompt": "Propose reliability-focused maintenance improvements.",
                    },
                ],
            ),
            LogicStep(
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

    result = workflow.run({}, request_id=request_id)
    store.close()

    payload = {
        "example": "workflow/workflow_delegate_and_memory_steps.py",
        "success": result.success,
        "execution_order": list(result.execution_order),
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
