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

The API is constructor-first: agents and workflows expose customization through
``__init__`` kwargs (prompt overrides, routing/tool allowlists, and workflow
run defaults). Workflow helper factory functions are intentionally not part of
the exported contract.


.. automodule:: design_research_agents
   :members:
   :undoc-members:
