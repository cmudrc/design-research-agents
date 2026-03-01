GroqServiceLLMClient
====================

``GroqServiceLLMClient`` targets the official Groq API via the ``groq`` SDK.

Default behavior
----------------

- Default model: ``llama-3.1-8b-instant``
- Remote execution through Groq service endpoints

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import GroqServiceLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = GroqServiceLLMClient()
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Give one concise architecture risk."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Install provider SDK extra: ``pip install -e ".[groq]"``
- ``GROQ_API_KEY`` (or pass ``api_key`` directly)
- Network access to Groq API

Examples
--------

- ``examples/clients/groq_service_client.py``

Attribution
-----------

- Docs: `Groq Python SDK <https://github.com/groq/groq-python>`_
- Console + API docs: `Groq docs portal <https://console.groq.com/docs/overview>`_
