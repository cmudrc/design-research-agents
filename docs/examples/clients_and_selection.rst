Clients and Model Selection Examples
====================================

These scripts run one representative client call plus model-selection policy
outcomes for design-oriented prompts.

Client configuration scripts
----------------------------

- ``examples/clients/llama_cpp_server_client.py``
  Public API: ``LlamaCppServerLLMClient``.
- ``examples/clients/openai_service_client.py``
  Public API: ``OpenAIServiceLLMClient``.
- ``examples/clients/openai_compatible_http_client.py``
  Public API: ``OpenAICompatibleHTTPLLMClient``.
- ``examples/clients/transformers_local_client.py``
  Public API: ``TransformersLocalLLMClient``.
- ``examples/clients/mlx_local_client.py``
  Public API: ``MlxLocalLLMClient``.

Observe: ``llm_call.response_has_text``, backend/capability fields, and
``trace.trace_path``.

Model selection scripts
-----------------------

- ``examples/model_selection/local.py``
- ``examples/model_selection/remote.py``

Public API: ``ModelSelector``.
Observe: decision payload (provider/model/rationale) and trace metadata.

Observed local run snippets (2026-02-21)
----------------------------------------

``examples/clients/transformers_local_client.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/clients/transformers_local_client.py

Observed stdout:

.. code-block:: json

   {
     "client_class": "TransformersLocalLLMClient",
     "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
     "llm_call": {
       "execution_mode": "deterministic_stub",
       "response_has_text": true,
       "response_provider": "deterministic"
     },
     "trace": {
       "trace_path": "artifacts/examples/traces/run_<timestamp>_example-clients-transformers-local-call-001.jsonl"
     }
   }

``examples/model_selection/local.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/model_selection/local.py

Observed stdout:

.. code-block:: json

   {
     "example": "model_selection/local.py",
     "provider": "llama_cpp",
     "model_id": "qwen3-14b-instruct-gguf-q4_k_m",
     "policy_id": "default",
     "safety_constraints": {"max_cost_usd": 0.01, "max_latency_ms": null}
   }
