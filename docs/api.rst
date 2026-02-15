API
===

Public API Contract
-------------------

The supported top-level API is the curated export list from
``design_research_agents.__all__``:

- Clients and Models
    - ``LlamaCppServerLLMClient``
    - ``OpenAIServiceLLMClient``
    - ``OpenAICompatibleHTTPLLMClient``
    - ``TransformersLocalLLMClient``
    - ``MlxLocalLLMClient``
    - ``ModelSelector``
- Tools
    - ``Toolbox``
    - ``CallableTool``
    - ``ScriptTool``
    - ``McpServer``
- Agents
    - ``SingleStepDirectLLMAgent``
    - ``SingleStepRouterAgent``
    - ``SingleStepJsonToolCallingAgent``
    - ``SingleStepCodeToolCallingAgent``
    - ``MultiStepJsonToolCallingAgent``
    - ``MultiStepCodeToolCallingAgent``
- Workflows
    - ``PlanExecuteWorkflow``
    - ``ProposeAndCritiqueWorkflow``
    - ``AgentRoutingWorkflow``
    - ``PureToolWorkflow``
    - ``MixedAgentWorkflow``


.. automodule:: design_research_agents
   :members:
   :undoc-members: