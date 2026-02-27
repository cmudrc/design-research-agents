API
===

This page documents the supported top-level public API from
``design_research_agents.__all__``.

Guaranteed compatibility applies to this top-level API surface and the public
facade modules documented in ``docs/reference`` under "Guaranteed Public
Modules".

Underscored module paths (for example ``design_research_agents._contracts``)
are internal and unstable. They are documented in module reference for
contributors but are not compatibility-guaranteed.

Top-level groups:

- Metadata: ``__version__``
- Entry points: agents, LLM clients, ``ModelSelector``
- Core contracts: ``ExecutionResult``, ``LLMRequest``, ``LLMMessage``, ``LLMResponse``, ``ToolResult``
- Orchestration: workflow step classes, ``Workflow``, and pattern classes
  (module homes: ``design_research_agents.workflow`` and
  ``design_research_agents.patterns``)
- Tools: ``Toolbox``, ``CallableToolConfig``, ``ScriptToolConfig``, ``MCPServerConfig``
- Tracing: ``Tracer``

``__version__``
---------------

.. autodata:: design_research_agents.__version__

Entry Points
------------

Agents
^^^^^^

.. autoclass:: design_research_agents.DirectLLMCall
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MultiStepAgent
   :members:
   :undoc-members:

LLM Clients and Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: design_research_agents.LlamaCppServerLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.GeminiServiceLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.GroqServiceLLMClient
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

.. autoclass:: design_research_agents.MLXLocalLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.VLLMServerLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.OllamaLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.SGLangServerLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ModelSelector
   :members:
   :undoc-members:

Core Contracts
^^^^^^^^^^^^^^

.. autoclass:: design_research_agents.ExecutionResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.LLMRequest
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.LLMMessage
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.LLMResponse
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.ToolResult
   :members:
   :undoc-members:
   :no-index:

Orchestration
-------------

Workflow Steps and Facade
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: design_research_agents.LogicStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ToolStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.AgentStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ModelStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.DelegateBatchStep
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

Patterns
^^^^^^^^

.. autoclass:: design_research_agents.ConversationPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.DebatePattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.PlanExecutePattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ReflexionPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.AgentRoutingPattern
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

.. autoclass:: design_research_agents.RAGPattern
   :members:
   :undoc-members:

Tools
-----

.. autoclass:: design_research_agents.Toolbox
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.CallableToolConfig
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ScriptToolConfig
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.MCPServerConfig
   :members:
   :undoc-members:

Tracing
-------

.. autoclass:: design_research_agents.Tracer
   :members:
   :undoc-members:
