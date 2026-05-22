Study
=====

``design_research_agents.study`` is the stable public facade for running agent
participants from study orchestration code and normalizing their outputs into a
shared envelope.

The older ``design_research_agents.integration`` module remains available for
compatibility. New study runners should prefer the typed request object and
facade exports documented here.

.. autoclass:: design_research_agents.study.AgentRunRequest

.. autoclass:: design_research_agents.study.AgentExecutionEnvelope

.. autoclass:: design_research_agents.study.StudyCondition

.. autofunction:: design_research_agents.study.execute_agent_request

.. autofunction:: design_research_agents.study.execute_agent_run

.. autofunction:: design_research_agents.study.normalize_agent_execution
