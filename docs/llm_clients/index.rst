LLM Clients
===========

The framework exposes constructor-first client classes that implement the
``LLMClient`` contract. Choose a client based on deployment constraints first,
then tune model selection.

Comparison matrix
-----------------

.. list-table::
   :header-rows: 1

   * - Client
     - Execution location
     - Default model behavior
     - Setup burden
     - Privacy / cost / latency profile
   * - ``LlamaCppServerLLMClient``
     - Local managed ``llama_cpp.server`` process
     - Defaults to ``api_model="qwen2-1.5b-q4"`` mapped to a local GGUF
     - Medium (local runtime + model download)
     - Strong privacy, lowest marginal cost, variable latency by hardware
   * - ``TransformersLocalLLMClient``
     - Local in-process transformers runtime
     - Defaults to ``model_id=default_model="distilgpt2"``
     - Medium-high (framework + model weights)
     - Strong privacy, lowest marginal cost, latency depends on device
   * - ``MlxLocalLLMClient``
     - Local Apple MLX runtime
     - Defaults to ``mlx-community/Qwen2.5-1.5B-Instruct-4bit``
     - Medium (Apple silicon + MLX stack)
     - Strong privacy, lowest marginal cost, strong local throughput on Apple hardware
   * - ``OpenAIServiceLLMClient``
     - Remote OpenAI API
     - Defaults to ``gpt-4o-mini``
     - Low (API key)
     - Lowest setup effort, network/data egress tradeoff, usage-based cost
   * - ``OpenAICompatibleHTTPLLMClient``
     - Remote or local OpenAI-compatible endpoint
     - Defaults to ``qwen2-1.5b-q4``
     - Low-medium (compatible server + endpoint config)
     - Flexible privacy/cost posture based on endpoint hosting

When to choose what
-------------------

1. Need strict data-local execution: start with ``LlamaCppServerLLMClient``,
   ``TransformersLocalLLMClient``, or ``MlxLocalLLMClient``.
2. Need fastest onboarding and hosted quality: use ``OpenAIServiceLLMClient``.
3. Need provider portability or self-hosted OpenAI-compatible infra: use
   ``OpenAICompatibleHTTPLLMClient``.
4. Need policy-driven choice between local and remote options: use
   :doc:`model_selection`.

See examples
------------

- ``examples/clients/README.md``

Pages
-----

- :doc:`model_selection`
- :doc:`llama_cpp_server`
- :doc:`openai_service`
- :doc:`openai_compatible_http`
- :doc:`transformers_local`
- :doc:`mlx_local`

.. toctree::
   :maxdepth: 2
   :hidden:

   model_selection
   llama_cpp_server
   openai_service
   openai_compatible_http
   transformers_local
   mlx_local
