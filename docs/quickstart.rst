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

   PYTHONPATH=src python3 examples/agents/basic/single_step_direct_llm_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_router_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_json_tool_calling_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_code_tool_calling_agent.py
   PYTHONPATH=src python3 examples/agents/basic/multi_step_code_tool_calling_agent.py
   PYTHONPATH=src python3 examples/agents/basic/multi_step_json_tool_calling_agent.py
   PYTHONPATH=src python3 examples/orchestrator/workflow_runtime.py
   PYTHONPATH=src python3 examples/orchestrator/plan_execute.py
   PYTHONPATH=src python3 examples/orchestrator/propose_critic.py
   PYTHONPATH=src python3 examples/orchestrator/agent_routing.py
   PYTHONPATH=src python3 examples/orchestrator/pure_tool_workflow.py
   PYTHONPATH=src python3 examples/orchestrator/mixed_agent_workflow.py
   PYTHONPATH=src python3 examples/model_selection/local.py
   PYTHONPATH=src python3 examples/model_selection/remote.py
   PYTHONPATH=src python3 examples/tools/mcp_minimal.py
   PYTHONPATH=src python3 examples/tools/source_fusion_story.py

Run additional streaming examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/streaming/single_step_direct_llm_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_router_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_json_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_code_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/multi_step_code_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/multi_step_json_tool_calling_agent_stream.py

Agent/orchestrator examples default to a local llama-cpp server via
``dra.llm.create_default_llm_client()``.

Use router-first backend configuration:

.. code-block:: python

   import design_research_agents as dra

   # Example YAML defines one or more configured backends.
   # See ``src/design_research_agents/llm/config.py`` for the schema.
   router = dra.llm.configure_router_from_yaml("configs/llm.yaml", default_backend="llama-local")

   # Optional: ``backend=...`` pins calls to a specific named backend.
   llm_client = dra.llm.BaseLLMClient(router=router, backend="llama-local")
   response = llm_client.chat(
       messages=[dra.contracts.llm.LLMMessage(role="user", content="Hello")],
       model=llm_client.default_model(),
       params=dra.contracts.llm.LLMChatParams(),
   )
   text = response.text

For contribution workflow and PR expectations, see ``CONTRIBUTING.md`` in the
repository root.
