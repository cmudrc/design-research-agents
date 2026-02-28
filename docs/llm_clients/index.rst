LLM Clients
===========

The framework exposes constructor-first client classes that implement the
``LLMClient`` contract. Choose a client based on deployment constraints first,
then tune model selection.

Every public client also exposes a shared introspection surface for debugging
and diagnostics: ``default_model()``, ``capabilities()``,
``config_snapshot()``, ``server_snapshot()``, and ``describe()``.

For install profiles and platform constraints for backend extras, see
:doc:`../dependencies_and_extras`.

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
   * - ``MLXLocalLLMClient``
     - Local Apple MLX runtime
     - Defaults to ``mlx-community/Qwen2.5-1.5B-Instruct-4bit``
     - Medium (Apple silicon + MLX stack)
     - Strong privacy, lowest marginal cost, strong local throughput on Apple hardware
   * - ``VLLMServerLLMClient``
     - Local/self-hosted vLLM OpenAI-compatible server
     - Defaults to ``api_model="qwen2.5-1.5b-instruct"``
     - Medium-high (server runtime + model provisioning)
     - Strong privacy, strong throughput, ops overhead for service management
   * - ``OllamaLLMClient``
     - Local/self-hosted Ollama daemon
     - Defaults to ``default_model="qwen2.5:1.5b-instruct"``
     - Low-medium (Ollama runtime + model pull)
     - Strong privacy, low setup burden, laptop-friendly local serving
   * - ``SGLangServerLLMClient``
     - Local/self-hosted SGLang OpenAI-compatible server
     - Defaults to ``model="Qwen/Qwen2.5-1.5B-Instruct"``
     - Medium-high (server runtime + model provisioning)
     - Strong privacy, high throughput, ops overhead for service management
   * - ``OpenAIServiceLLMClient``
     - Remote OpenAI API
     - Defaults to ``gpt-4o-mini``
     - Low (API key)
     - Lowest setup effort, network/data egress tradeoff, usage-based cost
   * - ``AzureOpenAIServiceLLMClient``
     - Remote Azure OpenAI API
     - Defaults to ``gpt-4o-mini`` (deployment name)
     - Low-medium (API key + endpoint + API version)
     - Azure-native hosted option, network/data egress tradeoff, Azure-managed billing
   * - ``AnthropicServiceLLMClient``
     - Remote Anthropic API
     - Defaults to ``claude-3-5-haiku-latest``
     - Low (API key)
     - Low setup effort, network/data egress tradeoff, usage-based cost
   * - ``GeminiServiceLLMClient``
     - Remote Gemini API
     - Defaults to ``gemini-2.5-flash``
     - Low (API key)
     - Low setup effort, network/data egress tradeoff, usage-based cost
   * - ``GroqServiceLLMClient``
     - Remote Groq API
     - Defaults to ``llama-3.1-8b-instant``
     - Low (API key)
     - Low setup effort, network/data egress tradeoff, usage-based cost
   * - ``OpenAICompatibleHTTPLLMClient``
     - Remote or local OpenAI-compatible endpoint
     - Defaults to ``qwen2-1.5b-q4``
     - Low-medium (compatible server + endpoint config)
     - Flexible privacy/cost posture based on endpoint hosting

When to choose what
-------------------

1. Need strict data-local execution: start with ``LlamaCppServerLLMClient``,
   ``TransformersLocalLLMClient``, ``MLXLocalLLMClient``, ``VLLMServerLLMClient``,
   ``OllamaLLMClient``, or ``SGLangServerLLMClient``.
2. Need fastest onboarding and hosted quality: use ``OpenAIServiceLLMClient``,
   ``AzureOpenAIServiceLLMClient``, ``AnthropicServiceLLMClient``,
   ``GeminiServiceLLMClient``, or ``GroqServiceLLMClient``.
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
- :doc:`azure_openai_service`
- :doc:`anthropic_service`
- :doc:`gemini_service`
- :doc:`groq_service`
- :doc:`openai_compatible_http`
- :doc:`transformers_local`
- :doc:`mlx_local`
- :doc:`vllm_server`
- :doc:`ollama_local`
- :doc:`sglang_server`

.. toctree::
   :maxdepth: 2
   :hidden:

   model_selection
   llama_cpp_server
   openai_service
   azure_openai_service
   anthropic_service
   gemini_service
   groq_service
   openai_compatible_http
   transformers_local
   mlx_local
   vllm_server
   ollama_local
   sglang_server
