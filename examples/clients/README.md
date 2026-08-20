## Client Call Examples

These scripts run one traced representative LLM call per public client class
and emit configuration plus call metadata.

## Scripts

- `llama_cpp_server_client.py`
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

These checkout-only commands reproduce the documented outputs without API
credentials, model downloads, or live services:

```bash
export PYTHONPATH=tests/example_monkeypatch:src
export DRA_EXAMPLE_LLM_MODE=deterministic
python examples/clients/llama_cpp_server_client.py
python examples/clients/anthropic_service_client.py
python examples/clients/gemini_service_client.py
python examples/clients/groq_service_client.py
python examples/clients/openai_service_client.py
python examples/clients/openai_compatible_http_client.py
python examples/clients/transformers_local_client.py
python examples/clients/mlx_local_client.py
python examples/clients/vllm_server_client.py
python examples/clients/ollama_local_client.py
python examples/clients/sglang_server_client.py
```

For a real run, unset `DRA_EXAMPLE_LLM_MODE` and follow the owning setup page
for the required extra, credentials, service, or local model:

- [llama.cpp server](../../docs/llm_clients/llama_cpp_server.rst)
- [Anthropic](../../docs/llm_clients/anthropic_service.rst)
- [Gemini](../../docs/llm_clients/gemini_service.rst)
- [Groq](../../docs/llm_clients/groq_service.rst)
- [OpenAI](../../docs/llm_clients/openai_service.rst)
- [OpenAI-compatible HTTP](../../docs/llm_clients/openai_compatible_http.rst)
- [Transformers](../../docs/llm_clients/transformers_local.rst)
- [MLX](../../docs/llm_clients/mlx_local.rst)
- [vLLM](../../docs/llm_clients/vllm_server.rst)
- [Ollama](../../docs/llm_clients/ollama_local.rst)
- [SGLang](../../docs/llm_clients/sglang_server.rst)

## Expected Outputs

- JSON payloads containing ``llm_call`` with response fields and execution mode.
- Trace metadata included in each payload.
