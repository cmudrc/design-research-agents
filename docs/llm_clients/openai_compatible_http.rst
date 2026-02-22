OpenAICompatibleHTTPLLMClient
=============================

``OpenAICompatibleHTTPLLMClient`` targets any OpenAI-compatible HTTP endpoint.

Default behavior
----------------

- Default base URL: ``http://127.0.0.1:8001/v1``
- Default model: ``qwen2-1.5b-q4``
- Execution locality depends on where your compatible endpoint is hosted

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import OpenAICompatibleHTTPLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = OpenAICompatibleHTTPLLMClient(base_url="http://127.0.0.1:8001/v1")
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Draft a one-line abstract."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Reachable OpenAI-compatible HTTP endpoint
- Optional API key env variable depending on upstream server behavior

Examples
--------

- ``examples/clients/openai_compatible_http_client.py``

Attribution
-----------

- Docs: `OpenAI API compatibility target <https://developers.openai.com/api/reference/overview>`_
- Homepage: `OpenAI developer portal <https://developers.openai.com/>`_
