GeminiServiceLLMClient
======================

``GeminiServiceLLMClient`` targets the official Gemini API via ``google-genai``.

Default behavior
----------------

- Default model: ``gemini-2.5-flash``
- Remote execution through Gemini service endpoints

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import GeminiServiceLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = GeminiServiceLLMClient()
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="List three design critique heuristics."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Install provider SDK extra: ``pip install -e ".[gemini]"``
- ``GOOGLE_API_KEY`` (fallback to ``GEMINI_API_KEY`` when unset)
- Network access to Gemini API

Examples
--------

- ``examples/clients/gemini_service_client.py``

Attribution
-----------

- Docs: `Google Gen AI Python SDK <https://googleapis.github.io/python-genai/>`_
- API key setup: `Gemini API key docs <https://ai.google.dev/gemini-api/docs/api-key>`_
