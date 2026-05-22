API
===

This page documents the supported top-level public API from
``design_research_agents.__all__``.

Guaranteed compatibility applies to this top-level API surface and to the
public facade modules documented in ``docs/reference`` under "Guaranteed
Public Modules".

Underscored module paths (for example ``design_research_agents._contracts``)
are internal and unstable. They are documented in the module reference for
contributors, but they are not compatibility-guaranteed.

Top-level groups:

- Metadata: ``__version__``
- Entry points: agents, study execution helpers, LLM clients, ``ModelSelector``, and model flights/catalogs
- Skills: ``SkillsConfig``
- Core contracts: ``ExecutionResult``, ``LLMRequest``, ``LLMMessage``, ``LLMResponse``, ``ToolResult``
  with normalized read helpers for structured payload access
- Orchestration: workflow step classes, ``Workflow``, workflow builders,
  and pattern classes
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

.. autoclass:: design_research_agents.SeededRandomBaselineAgent
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.PromptWorkflowAgent
   :members:
   :undoc-members:

Study Execution
^^^^^^^^^^^^^^^

``design_research_agents.study`` provides the stable public facade for
experiment runners. ``design_research_agents.integration`` remains available as
a compatibility module.

.. autoclass:: design_research_agents.AgentRunRequest
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.AgentExecutionEnvelope
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.StudyCondition
   :members:
   :undoc-members:
   :no-index:

.. autofunction:: design_research_agents.execute_agent_request

.. autofunction:: design_research_agents.execute_agent_run

.. autofunction:: design_research_agents.normalize_agent_execution

Skills
^^^^^^

.. autoclass:: design_research_agents.SkillsConfig
   :members:
   :undoc-members:

LLM Clients, Flights, and Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All public LLM clients implement the same introspection helpers in addition to
generation methods: ``default_model()``, ``capabilities()``,
``config_snapshot()``, ``server_snapshot()``, and ``describe()``. They also
implement ``close()`` plus ``with``-statement lifecycle support; the
context-manager form is the preferred public usage pattern.

.. autoclass:: design_research_agents.LlamaCppServerLLMClient
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.AnthropicServiceLLMClient
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

.. autoclass:: design_research_agents.AzureOpenAIServiceLLMClient
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

.. autoclass:: design_research_agents.ModelCatalog
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.ModelFlight
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.ModelFlightRegistry
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: design_research_agents.ModelSelector
   :members:
   :undoc-members:
   :no-index:

Core Contracts
^^^^^^^^^^^^^^

``ExecutionResult`` and per-step ``WorkflowStepResult`` objects expose matching
output access helpers for safe reads from loosely structured payloads. The
public ``ToolResult`` contract also includes normalized getters such as
``result_dict()``, ``result_list()``, ``error_message``, and ``artifact_paths``.

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

``CompiledExecution`` is the workflow-backed object returned by delegate
``compile(...)`` methods. Calling ``compiled.run()`` executes the bound
workflow and applies delegate-specific finalization. Accessing
``compiled.workflow`` gives the raw workflow graph for inspection and testing.
Calling ``compiled.workflow.run(...)`` directly bypasses that finalization
layer and returns the raw workflow result. Use ``compiled.to_mermaid()`` /
``compiled.to_svg()`` for direct compiled-workflow diagrams, or
``delegate.compile_to_mermaid()`` / ``delegate.compile_to_svg()`` to render
the most recently compiled workflow stored on a delegate instance.

Workflow step executions surface ``WorkflowStepResult`` payloads through
``ExecutionResult.step_results``. These step results mirror the top-level
``ExecutionResult`` output accessor helpers for consistent reads.

.. autoclass:: design_research_agents.LogicStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ToolStep
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.DelegateStep
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

.. autoclass:: design_research_agents.CompiledExecution
   :members:
   :undoc-members:

.. autofunction:: design_research_agents.build_json_prompt_workflow

Patterns
^^^^^^^^

Pattern ``compile(...)`` methods are the lower-level construction hook for
advanced callers. They return a bound ``CompiledExecution`` and omit the
top-level ``run()`` convenience wrapper until you call ``compiled.run()``.

.. autoclass:: design_research_agents.TwoSpeakerConversationPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.DebatePattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.PlanExecutePattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ProposeCriticPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.RalphLoopPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.NominalTeamPattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.RouterDelegatePattern
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.RoundBasedCoordinationPattern
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

.. autoclass:: design_research_agents.SimulatedAnnealingPattern
   :members:
   :undoc-members:

Temperature Schedules
^^^^^^^^^^^^^^^^^^^^^

Temperature schedules control the cooling rate during simulated annealing.
Pass an instance to ``SimulatedAnnealingPattern`` via the
``temperature_schedule`` parameter. Subclass ``TemperatureSchedule`` to
implement custom schedules.

.. autoclass:: design_research_agents.TemperatureSchedule
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.LinearSchedule
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.ExponentialSchedule
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.LogarithmicSchedule
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.AdaptiveSchedule
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
