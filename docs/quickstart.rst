Quickstart
==========

Requires Python 3.12+ and assumes you are working from the repository root.

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

Path A: Built-In OpenAI-Compatible HTTP
---------------------------------------

Use this when you want the default built-in client with no provider SDKs. It
works with any OpenAI-compatible endpoint, including local servers such as
llama.cpp, vLLM, and SGLang, or a remote compatibility gateway.

1. Install only the base package:

.. code-block:: bash

   pip install -e .

2. Point the client at your endpoint:

.. code-block:: python

   from design_research_agents import DirectLLMCall, OpenAICompatibleHTTPLLMClient

   with OpenAICompatibleHTTPLLMClient(
       base_url="http://127.0.0.1:8001/v1",
       default_model="qwen2-1.5b-q4",
   ) as llm_client:
       agent = DirectLLMCall(llm_client=llm_client)
       result = agent.run("List three interview themes about onboarding friction.")
       print(result.output)

Path B: Cloud (OpenAI)
----------------------

Use this when you want the fastest managed hosted path for a typical first run.

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

Path C: Local (llama.cpp recommended)
-------------------------------------

Use this when you want the primary managed local path.

1. Install backend-specific extras for local inference:

.. code-block:: bash

   pip install -e ".[dev,llama_cpp]"      # managed llama.cpp server client
   # or: pip install -e ".[dev,transformers]"  # in-process transformers backend
   # or: pip install -e ".[dev,mlx]"           # Apple MLX backend
   # or: pip install -e ".[dev,local]"         # core local backends
   # or: pip install -e ".[dev,full]"          # local + Linux server backends

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
