AzureOpenAIServiceLLMClient
===========================

``AzureOpenAIServiceLLMClient`` targets Azure OpenAI via the official ``openai`` SDK.

Default behavior
----------------

- Default model: ``gpt-4o-mini`` (used as the Azure deployment name)
- Remote execution through Azure OpenAI service endpoints

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import AzureOpenAIServiceLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = AzureOpenAIServiceLLMClient(
       azure_endpoint="https://your-resource.openai.azure.com",
       api_version="2025-01-01-preview",
   )
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Give me three study themes."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- ``AZURE_OPENAI_API_KEY`` (or pass ``api_key`` directly)
- ``AZURE_OPENAI_ENDPOINT`` (or pass ``azure_endpoint`` directly)
- ``AZURE_OPENAI_API_VERSION`` (or pass ``api_version`` directly)
- Network access to Azure OpenAI

Attribution
-----------

- Docs: `Azure OpenAI REST API reference <https://learn.microsoft.com/azure/ai-services/openai/reference>`_
- Homepage: `Azure OpenAI overview <https://learn.microsoft.com/azure/ai-services/openai/overview>`_
