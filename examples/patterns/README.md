## Pattern Examples

These scripts demonstrate reusable orchestration patterns built on top of workflow primitives.

## Scripts

- `plan_execute.py`
- `propose_critic.py`
- `agent_routing.py`
- `debate_pattern.py`
- `conversation_pattern.py`
- `networked_blackboard.py`
- `tree_search.py`
- `rag_reasoning.py`

## Quick Start

```bash
PYTHONPATH=src python3 examples/patterns/plan_execute.py
PYTHONPATH=src python3 examples/patterns/agent_routing.py
PYTHONPATH=src python3 examples/patterns/networked_blackboard.py
```

## Expected Outputs

- Pattern-specific result envelopes with `final_output` and termination metadata.
- Trace metadata (`trace.request_id`, `trace.trace_path`) for each run.
