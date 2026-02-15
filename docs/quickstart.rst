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
   PYTHONPATH=src python3 examples/workflow/workflow_runtime.py
   PYTHONPATH=src python3 examples/workflow/plan_execute.py
   PYTHONPATH=src python3 examples/workflow/propose_critic.py
   PYTHONPATH=src python3 examples/workflow/agent_routing.py
   PYTHONPATH=src python3 examples/workflow/pure_tool_workflow.py
   PYTHONPATH=src python3 examples/workflow/mixed_agent_workflow.py
   PYTHONPATH=src python3 examples/model_selection/local.py
   PYTHONPATH=src python3 examples/model_selection/remote.py
   PYTHONPATH=src python3 examples/tools/mcp_minimal.py
   PYTHONPATH=src python3 examples/tools/source_fusion_story.py

Workflow run signatures in the two reusable workflow chunks:

- ``mixed_agent_workflow``: initialize once, then call ``run(prompt=...)``.
- Supply ``agents`` and ``steps`` at init; no built-in mixed-step builder.
- ``pure_tool_workflow``: initialize once, then call ``run(inputs=...)`` with
  user-defined tool/logic ``steps`` supplied at init.
- Optional ``input_schema`` can validate run inputs; step topology and any
  scenario-specific behavior are caller-owned.

Run additional streaming examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/streaming/single_step_direct_llm_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_router_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_json_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_code_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/multi_step_code_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/multi_step_json_tool_calling_agent_stream.py

Agent/workflow examples default to a local llama-cpp server via
``LlamaCppServerLLMClient()``.

Use constructor-first provider clients:

.. code-block:: python

   from design_research_agents import LlamaCppServerLLMClient
   from design_research_agents.contracts.llm import LLMChatParams, LLMMessage

   llm_client = LlamaCppServerLLMClient()
   response = llm_client.chat(
       messages=[LLMMessage(role="user", content="Hello")],
       model=llm_client.default_model(),
       params=LLMChatParams(),
   )
   text = response.text

For contribution workflow and PR expectations, see ``CONTRIBUTING.md`` in the
repository root.
