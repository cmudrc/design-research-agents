VLLMServerLLMClient
===================

``VLLMServerLLMClient`` targets local/self-hosted ``vLLM`` OpenAI-compatible
inference endpoints.

Default behavior
----------------

- Default managed mode: ``manage_server=True``
- Default startup model: ``Qwen/Qwen2.5-1.5B-Instruct``
- Default API model alias: ``qwen2.5-1.5b-instruct``
- Default managed endpoint: ``http://127.0.0.1:8002/v1``

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import VLLMServerLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   with VLLMServerLLMClient() as client:
       response = client.generate(
           LLMRequest(
               messages=(LLMMessage(role="user", content="Give one architecture tradeoff."),),
               model=client.default_model(),
           )
       )

Prefer the context-manager form so managed servers shut down deterministically.
``close()`` remains available for explicit lifecycle control.

Dependencies and environment
----------------------------

- Install vLLM extras for managed mode: ``python -m pip install "design-research-agents[vllm]"``
- For connect mode, point at an existing vLLM-compatible endpoint with
  ``manage_server=False`` and ``base_url=...``.

Examples
--------

- ``examples/clients/vllm_server_client.py``

Attribution
-----------

- Docs: `vLLM docs <https://docs.vllm.ai/>`_
- Homepage: `vLLM GitHub <https://github.com/vllm-project/vllm>`_
