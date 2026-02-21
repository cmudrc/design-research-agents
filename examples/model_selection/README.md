## Model Selection Examples

These traced examples show how `ModelSelector` chooses local or remote models
for engineering-design tasks.

## Scripts

- `local.py`
  - Tight cost constraints, local-first selection behavior.
- `remote.py`
  - Heavy-load profile, remote-favoring selection behavior.

## Quick Start

```bash
PYTHONPATH=src python3 examples/model_selection/local.py
PYTHONPATH=src python3 examples/model_selection/remote.py
```

## Expected Outputs

- Structured selection decision payloads (`provider`, `model_id`, `rationale`).
- Trace metadata for each selection run.
