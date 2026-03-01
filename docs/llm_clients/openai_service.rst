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
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = OpenAIServiceLLMClient()
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Give me three study themes."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Install extra: ``pip install -e ".[openai]"`` (or ``pip install "design-research-agents[openai]"``)
- For the hosted-provider bundle, use ``pip install -e ".[providers]"``. For the full hosted + local bundle, use ``pip install -e ".[full]"``.
- ``OPENAI_API_KEY`` (or pass ``api_key`` directly)
- Network access to OpenAI API

Examples
--------

- ``examples/clients/openai_service_client.py``

Attribution
-----------

- Docs: `OpenAI API reference <https://developers.openai.com/api/reference/overview>`_
- Homepage: `OpenAI developer portal <https://developers.openai.com/>`_
