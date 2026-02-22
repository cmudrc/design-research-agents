SglangServerLLMClient
=====================

``SglangServerLLMClient`` targets local/self-hosted SGLang OpenAI-compatible
inference endpoints.

Default behavior
----------------

- Default managed mode: ``manage_server=True``
- Default startup model: ``Qwen/Qwen2.5-1.5B-Instruct``
- Default managed endpoint: ``http://127.0.0.1:30000/v1``

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import SglangServerLLMClient
   from design_research_agents._contracts import LLMChatParams, LLMMessage

   client = SglangServerLLMClient()
   response = client.chat(
       messages=[LLMMessage(role="user", content="Give one architecture tradeoff.")],
       model=client.default_model(),
       params=LLMChatParams(),
   )

Dependencies and environment
----------------------------

- Install SGLang extras for managed mode: ``pip install -e ".[sglang]"``
- For connect mode, point at an existing SGLang-compatible endpoint with
  ``manage_server=False`` and ``base_url=...``.

Examples
--------

- ``examples/clients/sglang_server_client.py``

Attribution
-----------

- Docs: `SGLang docs <https://docs.sglang.ai/>`_
- Homepage: `SGLang GitHub <https://github.com/sgl-project/sglang>`_
