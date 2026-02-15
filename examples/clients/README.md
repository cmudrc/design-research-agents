## Client Configuration Examples

These examples focus on configuring each `LLMClient` implementation with
explicit constructor arguments.

## What Each Example Demonstrates

- `llama_cpp_server_client.py`
  - Full local llama-cpp server client configuration (model aliasing, startup tuning).
- `openai_service_client.py`
  - Full OpenAI service client configuration (model defaults, auth, retries, patterns).
- `openai_compatible_http_client.py`
  - Full OpenAI-compatible HTTP client configuration (endpoint, auth, capability routing).
- `transformers_local_client.py`
  - Full in-process Transformers client configuration (device, dtype, quantization, revision).
- `mlx_local_client.py`
  - Full Apple MLX client configuration (model id, quantization, retries, model patterns).

## Quick Start

Run from repository root:

```bash
PYTHONPATH=src python3 examples/clients/llama_cpp_server_client.py
PYTHONPATH=src python3 examples/clients/openai_service_client.py
PYTHONPATH=src python3 examples/clients/openai_compatible_http_client.py
PYTHONPATH=src python3 examples/clients/transformers_local_client.py
PYTHONPATH=src python3 examples/clients/mlx_local_client.py
```

## Expected Outputs

- Each script prints a JSON configuration snapshot.
- No script executes model inference; these are constructor/config examples.

## Troubleshooting

- Missing optional local dependencies:
  - Install local extras: `pip install -e '.[local]'`.
