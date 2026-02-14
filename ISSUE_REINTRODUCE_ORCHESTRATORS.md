# Reintroduce Workflow Orchestrators (Sequential + DAG)

## Summary
Orchestrator support was intentionally removed from the public surface to reduce complexity while the agent/tool/LLM contracts stabilize. We should add orchestrators back behind a clear, tested, and maintainable integration plan.

## Background
Removed on: February 14, 2026

Removed components:
- `src/design_research_agents/orchestrator/`
- `src/design_research_agents/contracts/orchestrator.py`
- `examples/base_orchestrator_sequential.py`
- `examples/base_orchestrator_dag.py`
- `docs/orchestrator_types.rst`
- `tests/test_end_to_end_base.py`
- Package exports and docs references for orchestrators

Why removed:
- Core agent interfaces are still evolving.
- Orchestrator execution semantics (dependency context shape, failure propagation, cycle handling) were not yet locked down as long-term API.
- Keeping unstable workflow APIs exposed increases maintenance and compatibility risk.

## Goals
- Reintroduce workflow orchestration with stable, explicit contracts.
- Support two execution strategies:
  - Sequential (linear dependency-aware execution)
  - DAG (topological execution with cycle detection)
- Keep dependency context payload predictable across strategies.
- Provide robust error behavior and deterministic ordering guarantees.
- Restore docs/examples/tests only after behavior is contract-tested.

## Non-Goals
- Distributed execution or remote workers
- Automatic retries/backoff policy in v1 reintroduction
- Persistent workflow state/checkpointing
- Parallel DAG execution in the first reintroduction pass (optional follow-up)

## Proposed API Surface
Reintroduce the following only after contracts are finalized:
- `design_research_agents.contracts.orchestrator`
  - `WorkflowNode`
  - `WorkflowPlan` (optional; include only if used)
  - `WorkflowNodeResult`
  - `Orchestrator` protocol
- `design_research_agents.orchestrator`
  - `SequentialOrchestrator`
  - `DagOrchestrator`
- `design_research_agents.__init__` exports for all of the above

## Execution Semantics (Must Define Before Merge)
1. Node input/context behavior:
   - Node `input` passed as-is to agent.
   - Node `context` merged with orchestrator-injected dependency payload.
2. Dependency payload shape:
   - Stable key name (`dependency_results`) and deterministic value format.
3. Ordering guarantees:
   - Sequential: input order.
   - DAG: deterministic topological order when multiple nodes are eligible.
4. Failure handling:
   - Define whether downstream nodes are skipped or receive failed dependency state.
   - Define exception vs structured failure behavior for missing dependencies.
5. Validation:
   - Duplicate node IDs rejected.
   - Unknown dependency IDs rejected.
   - Cycles rejected with actionable error text.

## Implementation Plan
1. Recreate orchestrator contracts:
   - Add `src/design_research_agents/contracts/orchestrator.py`.
   - Add protocol and dataclasses with docstrings.
2. Recreate implementations:
   - Add `src/design_research_agents/orchestrator/sequential.py`.
   - Add `src/design_research_agents/orchestrator/dag.py`.
   - Add `src/design_research_agents/orchestrator/__init__.py`.
3. Restore exports:
   - Update `src/design_research_agents/contracts/__init__.py`.
   - Update `src/design_research_agents/__init__.py`.
4. Restore and harden tests:
   - Re-add end-to-end orchestrator tests.
   - Add edge-case tests for duplicate IDs, unknown deps, and deterministic DAG order.
   - Add strict cycle detection tests.
5. Restore docs and examples:
   - Re-add orchestrator docs page.
   - Re-add sequential and DAG examples.
   - Re-link from README and quickstart.

## Acceptance Criteria
- `pytest` includes orchestrator contract and e2e tests, all passing.
- Public imports work:
  - `from design_research_agents import SequentialOrchestrator, DagOrchestrator`
  - `from design_research_agents.contracts import WorkflowNode, WorkflowNodeResult, Orchestrator`
- DAG cycle detection is explicitly tested and error message is clear.
- Dependency result payload shape is documented and validated in tests.
- README/docs examples run without code changes.

## Risks
- API churn in agent result metadata could break orchestrator context contracts.
- Ambiguous failure semantics can create downstream behavior surprises.
- DAG ordering instability can create flaky tests and hard-to-debug behavior.

## Open Questions
- Should failed dependencies block downstream nodes by default?
- Should DAG v1 execute serially for determinism, with parallelism deferred?
- Do we need `WorkflowPlan` immediately, or can it be added in a follow-up?
- Should orchestrators live in core package or behind an optional extra/module?

## Suggested Milestone
- Milestone: `v0.2.0`
- Label suggestions: `enhancement`, `orchestration`, `api`, `docs`, `tests`
