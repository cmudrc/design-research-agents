Quickstart
==========

Requires Python 3.12+.

Install and run from repository root:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Path A: Hosted (fastest)
------------------------

Use this when you want the shortest path to a working run.

1. Install development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

2. Set API key:

.. code-block:: bash

   export OPENAI_API_KEY="<your-key>"

3. Run one agent call:

.. code-block:: python

   from design_research_agents import DirectLLMCall, OpenAIServiceLLMClient

   with OpenAIServiceLLMClient() as llm_client:
       agent = DirectLLMCall(llm_client=llm_client)
       result = agent.run("List three interview themes about onboarding friction.")
       print(result.output)

Path B: Local (privacy-first)
-----------------------------

Use this when you want local execution and are willing to manage local runtime/model setup.

1. Install backend-specific extras for local inference:

.. code-block:: bash

   pip install -e ".[dev,llama_cpp]"      # managed llama.cpp server client
   # or: pip install -e ".[dev,transformers]"  # in-process transformers backend
   # or: pip install -e ".[dev,mlx]"           # Apple MLX backend
   # or: pip install -e ".[dev,full]"          # all optional backend extras

2. Run one agent call with the managed llama.cpp server client:

.. code-block:: python

   from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient

   with LlamaCppServerLLMClient() as llm_client:
       agent = DirectLLMCall(llm_client=llm_client)
       result = agent.run("Summarize this study brief in five bullets.")
       print(result.output)

Checks and Docs
---------------

.. code-block:: bash

   make test
   make docs-build

Next Steps
----------

- Optional dependency profiles and platform notes: :doc:`dependencies_and_extras`
- Scenario-driven examples and expected outputs: :doc:`examples/index`
- Explore runnable examples: ``examples/README.md``
- LLM client setup details: :doc:`llm_clients/index`
- Agent behavior tradeoffs: :doc:`agents/index`
- Workflow builder primitives: :doc:`workflows/index`
- Prebuilt workflow implementations: :doc:`patterns/index`
- Tool runtime and integrations: :doc:`tools/index`
