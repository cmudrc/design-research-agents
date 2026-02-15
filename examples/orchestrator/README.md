## Orchestrator Examples

Run from repo root:

```bash
PYTHONPATH=src python3 examples/orchestrator/sequential.py
PYTHONPATH=src python3 examples/orchestrator/dag.py
PYTHONPATH=src python3 examples/orchestrator/research_pipeline_dag.py
```

Notes:
- `sequential.py` shows dependency-aware ordered execution.
- `dag.py` shows deterministic topological execution with branch routing.
- `research_pipeline_dag.py` shows a toy research workflow graph.
