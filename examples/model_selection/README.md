## Model Selection Examples

These examples show how model-selection policy chooses local or remote models
based on constraints and intent.

## What Each Example Demonstrates

- `local.py`
  - Tight-cost / local-preference selection that stays on local capability.
- `remote.py`
  - Constraints that allow or prefer remote capability when appropriate.

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
