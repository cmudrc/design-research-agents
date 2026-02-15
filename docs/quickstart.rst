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

   PYTHONPATH=src python3 examples/agents/basic/router_agent.py
   PYTHONPATH=src python3 examples/agents/basic/direct_llm_agent.py
   PYTHONPATH=src python3 examples/agents/basic/tool_calling_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_code_agent.py
   PYTHONPATH=src python3 examples/agents/basic/multi_step_agent.py
   PYTHONPATH=src python3 examples/runtime/plan_execute.py
   PYTHONPATH=src python3 examples/runtime/propose_critic.py
   PYTHONPATH=src python3 examples/runtime/triage.py
   PYTHONPATH=src python3 examples/orchestrator/pure_tool_workflow.py
   PYTHONPATH=src python3 examples/orchestrator/mixed_agent_workflow.py
   PYTHONPATH=src python3 examples/model_selection/local.py
   PYTHONPATH=src python3 examples/model_selection/remote.py

Run additional streaming examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/streaming/direct_llm_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/router_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_code_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/multi_step_agent_stream.py

These streaming examples use deterministic in-script LLM stubs and do not need
an external model backend.

Use router-first backend configuration:

.. code-block:: python

   from design_research_agents.contracts.llm import LLMChatParams, LLMMessage
   from design_research_agents.llm import BaseLLMClient, configure_router_from_yaml

   # Example YAML defines one or more configured backends.
   # See ``src/design_research_agents/llm/config.py`` for the schema.
   router = configure_router_from_yaml("configs/llm.yaml", default_backend="llama-local")

   # Optional: ``backend=...`` pins calls to a specific named backend.
   llm_client = BaseLLMClient(router=router, backend="llama-local")
   response = llm_client.chat(
       messages=[LLMMessage(role="user", content="Hello")],
       model=llm_client.default_model(),
       params=LLMChatParams(),
   )
   text = response.text

For contribution workflow and PR expectations, see ``CONTRIBUTING.md`` in the
repository root.
