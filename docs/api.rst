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
- Orchestration: workflow step classes, ``Workflow``, and pattern classes
- Tools: ``Toolbox``, ``CallableTool``, ``ScriptTool``, ``McpServer``
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

.. autoclass:: design_research_agents.VllmServerLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.OllamaLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.SglangServerLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ModelSelector
   :members:
   :undoc-members:

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

Tracing
-------

.. autoclass:: design_research_agents.Tracer
   :members:
   :undoc-members:
