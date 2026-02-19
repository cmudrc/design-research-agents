API
===

The supported top-level API is the curated export list from
``design_research_agents.__all__``:

- Metadata
    - ``__version__``
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
- Agents (core implementations)
    - ``SingleStepDirectLLMAgent``
    - ``SingleStepToolRouterAgent``
    - ``SingleStepJsonToolCallingAgent``
    - ``SingleStepCodeToolCallingAgent``
    - ``MultiStepDirectLLMAgent``
    - ``MultiStepToolRouterAgent``
    - ``MultiStepJsonToolCallingAgent``
    - ``MultiStepCodeToolCallingAgent``
- Patterns
    - ``ConversationPattern``
    - ``DebatePattern``
    - ``PlannerExecutorPattern``
    - ``ReflexionPattern``
    - ``RouterPattern``
    - ``NetworkedPattern``
    - ``BlackboardPattern``
    - ``TreeSearchPattern``
    - ``RagReasoningPattern``
- Workflows
    - ``LogicStep``
    - ``ToolStep``
    - ``AgentStep``
    - ``LoopStep``
    - ``MemoryReadStep``
    - ``MemoryWriteStep``
    - ``Workflow``

The API is constructor-first: agents and workflows expose customization through
``__init__`` kwargs (prompt overrides, routing/tool allowlists, and workflow
run defaults). Workflow helper factory functions are intentionally not part of
the exported contract.

For module-level implementation details (including internal, non-stable modules),
see :doc:`reference/index`.

``__version__``
---------------

.. autodata:: design_research_agents.__version__

Clients and Models
------------------

.. autoclass:: design_research_agents.LlamaCppServerLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.OpenAIServiceLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.OpenAICompatibleHTTPLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.TransformersLocalLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MlxLocalLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ModelSelector
   :members:
   :undoc-members:

Tools
-----

.. autoclass:: design_research_agents.Toolbox
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.CallableTool
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ScriptTool
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.McpServer
   :members:
   :undoc-members:

Agents
------

.. autoclass:: design_research_agents.SingleStepDirectLLMAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.SingleStepToolRouterAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.SingleStepJsonToolCallingAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.SingleStepCodeToolCallingAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MultiStepDirectLLMAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MultiStepToolRouterAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MultiStepJsonToolCallingAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MultiStepCodeToolCallingAgent
   :members:
   :undoc-members:

Patterns
--------

.. autoclass:: design_research_agents.ConversationPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.DebatePattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.PlannerExecutorPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ReflexionPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.RouterPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.NetworkedPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.BlackboardPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.TreeSearchPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.RagReasoningPattern
   :members:
   :undoc-members:

Workflows
---------

.. autoclass:: design_research_agents.LogicStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ToolStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.AgentStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.LoopStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MemoryReadStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MemoryWriteStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.Workflow
   :members:
   :undoc-members:
