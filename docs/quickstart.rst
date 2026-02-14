Quickstart
==========

Install for development:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e ".[dev,local]"

Run checks:

.. code-block:: bash

   make lint
   make typecheck
   make test

Run example:

.. code-block:: bash

   make run-example

Configure OpenAI defaults once per run:

.. code-block:: python

   from design_research_agents import complete, configure_openai

   configure_openai(model="gpt-4o-mini")
   text = complete("Hello", backend="openai")

.. code-block:: python

   from design_research_agents import complete, configure_llama_cpp_server

   configure_llama_cpp_server(model="/path/to/model.gguf")
   # Or Hugging Face:
   # configure_llama_cpp_server(
   #     model="tinyllama.Q4_K_M.gguf",
   #     hf_model_repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
   # )
   text = complete("Hello", backend="llama-cpp-server")

For contribution workflow and PR expectations, see ``CONTRIBUTING.md`` in the
repository root.
