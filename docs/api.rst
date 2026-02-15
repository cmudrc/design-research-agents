API
===

Public API Contract
-------------------

The supported top-level API is the curated export list from
``design_research_agents.__all__``:

- ``UnifiedToolRuntime``
- ``SingleStepDirectLLMAgent``
- ``SingleStepRouterAgent``
- ``SingleStepJsonToolCallingAgent``
- ``SingleStepCodeToolCallingAgent``
- ``MultiStepJsonToolCallingAgent``
- ``MultiStepCodeToolCallingAgent``
- ``PlanExecuteWorkflow``
- ``ProposeAndCritiqueWorkflow``
- ``AgentRoutingWorkflow``
- ``PureToolWorkflow``
- ``MixedAgentWorkflow``
- ``LlamaCppServerLLMClient``
- ``OpenAIServiceLLMClient``
- ``OpenAICompatibleHTTPLLMClient``
- ``TransformersLocalLLMClient``
- ``MlxLocalLLMClient``
- ``TraceConfig``
- ``configure_tracing``
- ``StdioMcpServer``
- ``serve_stdio``
- ``HardwareProfile``
- ``ModelSelectionPolicy``
- ``ModelSelectionIntent``
- ``ModelSelectionConstraints``

Submodules remain importable for advanced users, but are internal unless
explicitly documented.

Package Entrypoint
------------------

.. automodule:: design_research_agents
   :members:
   :undoc-members:

Core Contracts
--------------

.. automodule:: design_research_agents.contracts
   :members:
   :undoc-members:
   :no-index:
