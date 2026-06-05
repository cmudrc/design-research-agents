## Client Call Examples

These scripts run one traced representative LLM call per public client class
and emit configuration plus call metadata.

## Scripts

- `llama_cpp_server_client.py`
- `demo_client.py`
- `anthropic_service_client.py`
- `gemini_service_client.py`
- `groq_service_client.py`
- `openai_service_client.py`
- `openai_compatible_http_client.py`
- `transformers_local_client.py`
- `mlx_local_client.py`
- `vllm_server_client.py`
- `ollama_local_client.py`
- `sglang_server_client.py`

## Quick Start

```bash
PYTHONPATH=src python3 examples/clients/llama_cpp_server_client.py
PYTHONPATH=src python3 examples/clients/demo_client.py
PYTHONPATH=src python3 examples/clients/anthropic_service_client.py
PYTHONPATH=src python3 examples/clients/gemini_service_client.py
PYTHONPATH=src python3 examples/clients/groq_service_client.py
PYTHONPATH=src python3 examples/clients/openai_service_client.py
PYTHONPATH=src python3 examples/clients/openai_compatible_http_client.py
PYTHONPATH=src python3 examples/clients/transformers_local_client.py
PYTHONPATH=src python3 examples/clients/mlx_local_client.py
PYTHONPATH=src python3 examples/clients/vllm_server_client.py
PYTHONPATH=src python3 examples/clients/ollama_local_client.py
PYTHONPATH=src python3 examples/clients/sglang_server_client.py
```

## Expected Outputs

- JSON payloads containing ``llm_call`` with response fields and execution mode.
- Trace metadata included in each payload.
