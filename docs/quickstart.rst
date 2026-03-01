Quickstart
==========

Requires Python 3.12+ and assumes you are working from the repository root.

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Path A: Cloud (OpenAI)
----------------------

Use this when you want the fastest hosted path for a typical first run.

1. Install the OpenAI extra:

.. code-block:: bash

   pip install -e ".[dev,openai]"

   # If you are using Azure OpenAI instead, install:
   # pip install -e ".[dev,azure]"

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

For Azure OpenAI, use ``AzureOpenAIServiceLLMClient`` and install
``.[dev,azure]``. The ``azure`` extra installs the same ``openai`` SDK as the
``openai`` extra, but makes the backend intent explicit in setup commands.

Path B: Local (llama.cpp recommended)
-------------------------------------

Use this when you want the primary local path and are willing to manage local runtime/model setup.

1. Install backend-specific extras for local inference:

.. code-block:: bash

   pip install -e ".[dev,llama_cpp]"      # managed llama.cpp server client
   # or: pip install -e ".[dev,transformers]"  # in-process transformers backend
   # or: pip install -e ".[dev,mlx]"           # Apple MLX backend
   # or: pip install -e ".[dev,local]"         # all local backends for this platform
   # or: pip install -e ".[dev,full]"          # hosted providers + local backends

2. Run one agent call with the managed llama.cpp server client:

.. code-block:: python

   from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient

   with LlamaCppServerLLMClient() as llm_client:
       agent = DirectLLMCall(llm_client=llm_client)
       result = agent.run("Summarize this study brief in five bullets.")
       print(result.output)

Path C: Minimal Base Install
----------------------------

Use this when you want a no-OpenAI, no-provider-SDK path for power users, offline
inspection, or CI smoke checks.

1. Install only the base package:

.. code-block:: bash

   pip install -e .

2. Run one agent call with the built-in stand-in client:

.. code-block:: python

   from design_research_agents import DirectLLMCall, HTMLLLMClient

   with HTMLLLMClient() as llm_client:
       agent = DirectLLMCall(llm_client=llm_client)
       result = agent.run("List three interview themes about onboarding friction.")
       print(result.output)

Checks and Docs
---------------

.. code-block:: bash

   make test
   make docs-check
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
