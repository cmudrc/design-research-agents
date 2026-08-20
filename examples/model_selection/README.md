## Model Selection Examples

These traced examples show how `ModelSelector` chooses local or remote models
for engineering-design tasks. The same model-selection package also exposes
`ModelFlightRegistry` for enumerating named model sets such as `qwen3-gguf`,
`gemma3-gguf`, `llama-gguf`, `mistral-gguf`, `phi-gguf`, `open-reasoning`,
`frontier-moe-open-weights`, `agentic-coding-open-weights`,
`vision-language-open-weights`, and `openai-api`.

## Scripts

- `local.py`
  - Tight cost constraints, local-first selection behavior.
- `remote.py`
  - Heavy-load profile, remote-favoring selection behavior.

## Quick Start

```bash
PYTHONPATH=src python examples/model_selection/local.py
PYTHONPATH=src python examples/model_selection/remote.py
```

## Expected Outputs

- Structured selection decision payloads (`provider`, `model_id`, `rationale`).
- Trace metadata for each selection run.
