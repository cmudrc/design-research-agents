VllmServerLLMClient
===================

``VllmServerLLMClient`` targets local/self-hosted ``vLLM`` OpenAI-compatible
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

   from design_research_agents import VllmServerLLMClient
   from design_research_agents._contracts import LLMChatParams, LLMMessage

   client = VllmServerLLMClient()
   response = client.chat(
       messages=[LLMMessage(role="user", content="Give one architecture tradeoff.")],
       model=client.default_model(),
       params=LLMChatParams(),
   )

Dependencies and environment
----------------------------

- Install vLLM extras for managed mode: ``pip install -e ".[vllm]"``
- For connect mode, point at an existing vLLM-compatible endpoint with
  ``manage_server=False`` and ``base_url=...``.

Examples
--------

- ``examples/clients/vllm_server_client.py``

Attribution
-----------

- Docs: `vLLM docs <https://docs.vllm.ai/>`_
- Homepage: `vLLM GitHub <https://github.com/vllm-project/vllm>`_
