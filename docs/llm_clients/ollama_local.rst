OllamaLLMClient
===============

``OllamaLLMClient`` targets local/self-hosted Ollama chat inference.

Default behavior
----------------

- Default managed mode: ``manage_server=True``
- Default model: ``qwen2.5:1.5b-instruct``
- Default managed endpoint: ``http://127.0.0.1:11434``

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import OllamaLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = OllamaLLMClient()
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Summarize one design principle."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Install and run Ollama locally (`ollama serve`) if using connect mode.
- Managed mode starts ``ollama serve`` automatically using the configured
  ``ollama_executable``.
- Optional model prefetch: ``auto_pull_model=True``.

Examples
--------

- ``examples/clients/ollama_local_client.py``

Attribution
-----------

- Docs: `Ollama API docs <https://github.com/ollama/ollama/blob/main/docs/api.md>`_
- Homepage: `Ollama <https://ollama.com/>`_
