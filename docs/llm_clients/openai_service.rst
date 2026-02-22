OpenAIServiceLLMClient
======================

``OpenAIServiceLLMClient`` targets the official OpenAI API.

Default behavior
----------------

- Default model: ``gpt-4o-mini``
- Remote execution through OpenAI service endpoints

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import OpenAIServiceLLMClient
   from design_research_agents._contracts import LLMChatParams, LLMMessage

   client = OpenAIServiceLLMClient()
   response = client.chat(
       messages=[LLMMessage(role="user", content="Give me three study themes.")],
       model=client.default_model(),
       params=LLMChatParams(),
   )

Dependencies and environment
----------------------------

- ``OPENAI_API_KEY`` (or pass ``api_key`` directly)
- Network access to OpenAI API

Examples
--------

- ``examples/clients/openai_service_client.py``

Attribution
-----------

- Docs: `OpenAI API reference <https://developers.openai.com/api/reference/overview>`_
- Homepage: `OpenAI developer portal <https://developers.openai.com/>`_
