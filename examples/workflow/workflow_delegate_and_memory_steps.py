"""Run traced workflow covering delegate batch and memory read/write step APIs.

Expected observations:
- ``MemoryWriteStep`` seeds design constraints.
- ``MemoryReadStep`` retrieves relevant context records.
- ``DelegateBatchStep`` runs multiple deterministic delegates.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

from pathlib import Path

from design_research_agents import (
    DelegateBatchStep,
    LogicStep,
    MemoryReadStep,
    MemoryWriteStep,
    Workflow,
)
from design_research_agents.contracts import DelegateBatchCall, MemoryWriteRecord
from design_research_agents.memory.stores.sqlite_store import SQLiteMemoryStore
from design_research_agents.shared.deterministic_design_helpers import FixedDesignPeerAgent
from design_research_agents.shared.example_support import make_tracer, print_json, trace_info


def main() -> None:
    """Execute memory and delegate-batch primitives in one traced workflow."""
    request_id = "example-workflow-delegate-memory-design-001"
    db_path = Path("artifacts/examples/workflow_delegate_and_memory.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    store = SQLiteMemoryStore(db_path=db_path)

    workflow = Workflow(
        tool_runtime=None,
        memory_store=store,
        tracer=make_tracer(),
        input_mode="schema",
        steps=[
            MemoryWriteStep(
                step_id="seed_constraints",
                namespace="design_constraints",
                records_builder=lambda _context: [
                    MemoryWriteRecord(
                        content="Constraint: reduce service time by at least 20 percent.",
                        metadata={"kind": "constraint"},
                    ),
                    MemoryWriteRecord(
                        content="Constraint: preserve ingress protection sealing.",
                        metadata={"kind": "constraint"},
                    ),
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
                    DelegateBatchCall(
                        call_id="manufacturing_peer",
                        delegate=FixedDesignPeerAgent(
                            messages=[
                                "Use captive screws with standardized head type for faster "
                                "maintenance."
                            ]
                        ),
                        prompt=(
                            "Propose manufacturing-friendly maintenance improvements using "
                            "retrieved constraints count="
                            f"{context['dependency_results']['read_constraints']['output']['count']}."
                        ),
                    ),
                    DelegateBatchCall(
                        call_id="reliability_peer",
                        delegate=FixedDesignPeerAgent(
                            messages=[
                                "Add gasket alignment features to preserve ingress protection "
                                "after service."
                            ]
                        ),
                        prompt="Propose reliability-focused maintenance improvements.",
                    ),
                ],
            ),
            LogicStep(
                step_id="finalize",
                dependencies=("read_constraints", "peer_batch"),
                handler=lambda context: {
                    "constraints_found": (
                        context["dependency_results"]["read_constraints"]["output"]["count"]
                    ),
                    "delegate_calls": len(
                        context["dependency_results"]["peer_batch"]["output"].get("results", [])
                    ),
                    "final_delegate_output": (
                        context["dependency_results"]["peer_batch"]["output"].get("final_output")
                    ),
                },
            ),
        ],
    )

    result = workflow.run({}, request_id=request_id)
    store.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "example": "workflow/workflow_delegate_and_memory_steps.py",
        "success": result.success,
        "execution_order": list(result.execution_order),
        "final_output": output.get("final_output"),
        "error": output.get("error"),
        "trace": trace_info(request_id),
    }
    print_json(payload)


if __name__ == "__main__":
    main()
