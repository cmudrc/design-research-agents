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

Run additional contract-focused examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/router_agent.py
   PYTHONPATH=src python3 examples/direct_llm_agent.py
   PYTHONPATH=src python3 examples/tool_calling_agent.py
   PYTHONPATH=src python3 examples/single_step_code_agent.py
   PYTHONPATH=src python3 examples/multi_step_agent.py

Run additional streaming examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/direct_llm_agent_stream.py
   PYTHONPATH=src python3 examples/router_agent_stream.py
   PYTHONPATH=src python3 examples/tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/single_step_code_agent_stream.py
   PYTHONPATH=src python3 examples/multi_step_agent_stream.py

These streaming examples use deterministic in-script LLM stubs and do not need
an external model backend.

Use the llama-cpp backend directly:

.. code-block:: python

   from design_research_agents import complete, configure_llama_cpp_server

   configure_llama_cpp_server(
       model="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
       hf_model_repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
       api_model="tinyllama-q4-example",
   )
   text = complete("Hello")

For contribution workflow and PR expectations, see ``CONTRIBUTING.md`` in the
repository root.
