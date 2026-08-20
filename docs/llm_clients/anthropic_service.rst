AnthropicServiceLLMClient
=========================

``AnthropicServiceLLMClient`` targets the official Anthropic API via the ``anthropic`` SDK.

Default behavior
----------------

- Default model: ``claude-3-5-haiku-latest``
- Remote execution through Anthropic service endpoints

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import AnthropicServiceLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = AnthropicServiceLLMClient()
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Give one concise architecture risk."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Install provider SDK extra: ``python -m pip install "design-research-agents[anthropic]"``
- ``ANTHROPIC_API_KEY`` (or pass ``api_key`` directly)
- Network access to Anthropic API

Examples
--------

- ``examples/clients/anthropic_service_client.py``

Attribution
-----------

- Docs: `Anthropic API docs <https://platform.claude.com/docs/en/api/overview>`_
- SDK: `Anthropic Python SDK repository <https://github.com/anthropics/anthropic-sdk-python>`_
