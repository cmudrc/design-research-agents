## Pattern Examples

These scripts demonstrate reusable orchestration patterns built on top of workflow primitives.

## Scripts

- `plan_execute.py`
- `propose_critic.py`
- `router_delegate.py`
- `debate_pattern.py`
- `two_speaker_conversation.py`
- `coordination_patterns.py`
- `tree_search.py`
- `ralph_loop.py`
- `nominal_team.py`
- `rag.py`
- `simulated_annealing.py`

## Quick Start

```bash
PYTHONPATH=src python examples/patterns/plan_execute.py
PYTHONPATH=src python examples/patterns/router_delegate.py
PYTHONPATH=src python examples/patterns/coordination_patterns.py
```

## Expected Outputs

- Pattern-specific result envelopes with `final_output` and termination metadata.
- Trace metadata (`trace.request_id`, `trace.trace_path`) for each run.
