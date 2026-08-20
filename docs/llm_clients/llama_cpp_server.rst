LlamaCppServerLLMClient
=======================

``LlamaCppServerLLMClient`` runs a managed local ``llama_cpp.server`` process.

Default behavior
----------------

- Default GGUF artifact: ``Qwen2.5-1.5B-Instruct-Q4_K_M.gguf``
- Default API model name exposed to requests: ``qwen2-1.5b-q4``
- Local execution (no hosted API requirement)

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import LlamaCppServerLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   with LlamaCppServerLLMClient() as client:
       response = client.generate(
           LLMRequest(
               messages=(LLMMessage(role="user", content="Summarize this paragraph."),),
               model=client.default_model(),
           )
       )

Prefer the context-manager form so the managed local server always shuts down
deterministically. ``close()`` remains available for explicit lifecycle control.

Dependencies and environment
----------------------------

- Install llama.cpp backend extras: ``python -m pip install "design-research-agents[llama_cpp]"``
- Ensure local model download/runtime prerequisites are available.

Model notes for local runs
--------------------------

- Smaller quantized GGUF models (for example 1B-3B 4-bit) are best for fast
  iteration on laptops.
- Increase ``context_window`` and model size only when your RAM/latency budget
  supports it.
- Use :doc:`model_selection` to enforce local-only behavior plus cost/latency
  constraints consistently across workflows.

Examples
--------

- ``examples/clients/llama_cpp_server_client.py``

Attribution
-----------

- Docs: `llama.cpp server usage <https://github.com/ggml-org/llama.cpp/tree/master/tools/server>`_
- Homepage: `llama.cpp GitHub <https://github.com/ggml-org/llama.cpp>`_
