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
   PYTHONPATH=src python3 examples/agents/basic/single_step_tool_router_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_router_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_json_tool_calling_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_json_callable_tool_agent.py
   PYTHONPATH=src python3 examples/agents/basic/single_step_code_tool_calling_agent.py
   PYTHONPATH=src python3 examples/agents/basic/multi_step_direct_llm_agent.py
   PYTHONPATH=src python3 examples/agents/basic/multi_step_tool_router_agent.py
   PYTHONPATH=src python3 examples/agents/basic/multi_step_code_tool_calling_agent.py
   PYTHONPATH=src python3 examples/agents/basic/multi_step_json_tool_calling_agent.py
   PYTHONPATH=src python3 examples/workflow/workflow_runtime.py
   PYTHONPATH=src python3 examples/workflow/workflow_runtime_loop_step.py
   PYTHONPATH=src python3 examples/workflow/plan_execute.py
   PYTHONPATH=src python3 examples/workflow/propose_critic.py
   PYTHONPATH=src python3 examples/workflow/agent_routing.py
   PYTHONPATH=src python3 examples/workflow/debate_pattern.py
   PYTHONPATH=src python3 examples/workflow/workflow_schema_mode.py
   PYTHONPATH=src python3 examples/workflow/workflow_prompt_mode.py
   PYTHONPATH=src python3 examples/model_selection/local.py
   PYTHONPATH=src python3 examples/model_selection/remote.py
   PYTHONPATH=src python3 examples/tools/mcp_minimal.py
   PYTHONPATH=src python3 examples/tools/source_fusion_story.py
   PYTHONPATH=src python3 examples/tools/script_tools/python/single_step_json_script_rubric_score_agent.py
   PYTHONPATH=src python3 examples/tools/script_tools/bash/single_step_json_script_repo_quickscan_agent.py

Run client configuration examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/llama_cpp_server_client.py
   PYTHONPATH=src python3 examples/clients/openai_service_client.py
   PYTHONPATH=src python3 examples/clients/openai_compatible_http_client.py
   PYTHONPATH=src python3 examples/clients/transformers_local_client.py
   PYTHONPATH=src python3 examples/clients/mlx_local_client.py

Workflow run signatures in the reusable ``Workflow`` class:

- ``Workflow(input_mode='prompt')``: initialize once, then call ``run(\"...\")``.
- ``Workflow(input_mode='schema')``: initialize once, then call
  ``run({...})`` with optional ``input_schema`` validation.
- Supply ``steps`` at init and optionally ``agents`` when ``AgentStep`` entries
  are present in the graph.
- Step topology and any scenario-specific behavior are caller-owned.

Run additional streaming examples:

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/streaming/single_step_direct_llm_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_tool_router_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_router_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_json_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/single_step_code_tool_calling_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/multi_step_direct_llm_agent_stream.py
   PYTHONPATH=src python3 examples/agents/streaming/multi_step_tool_router_agent_stream.py
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

Continue with focused guides:

- :doc:`llm_clients/index`
- :doc:`agents/index`
- :doc:`tools/index`
- :doc:`workflows/index`

For contribution workflow and PR expectations, see
`CONTRIBUTING.md <https://github.com/cmudrc/design-research-agents/blob/main/CONTRIBUTING.md>`_.
