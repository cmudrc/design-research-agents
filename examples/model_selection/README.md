## Model Selection Examples

These examples show how ``ModelSelector`` chooses local or remote models
from flat selection inputs.

## What Each Example Demonstrates

- `local.py`
  - Tight-cost selection that tends to stay on local capability.
- `remote.py`
  - Constraints that allow or prefer remote capability when appropriate.

## Programmatic Outputs

``ModelSelector.select(...)`` supports:

- ``output="decision"`` for a structured selection decision.
- ``output="client_config"`` for a plain config mapping
  (``provider``, ``model_id``, ``client_class``, ``kwargs``, ...).
- ``output="client"`` (default) for an instantiated LLM client.

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/model_selection/local.py
PYTHONPATH=src python3 examples/model_selection/remote.py
```

## Expected Outputs

- Each script prints the selected model decision and associated rationale.
- Output should clearly indicate local-vs-remote selection behavior.

## Troubleshooting

- Unexpected selection decisions:
  - Check your hardware profile and constraint values in the script inputs.
