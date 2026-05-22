Model Selection
===============

``design_research_agents.model_selection`` is the stable public facade for model
catalogs, model flights, hardware snapshots, and selector decisions.

Public Facade
-------------

.. autoclass:: design_research_agents.model_selection.HardwareProfile

.. autoclass:: design_research_agents.model_selection.ModelCatalog

.. autoclass:: design_research_agents.model_selection.ModelCostHint

.. autoclass:: design_research_agents.model_selection.ModelFlight

.. autoclass:: design_research_agents.model_selection.ModelFlightRegistry

.. autoclass:: design_research_agents.model_selection.ModelLatencyHint

.. autoclass:: design_research_agents.model_selection.ModelMemoryHint

.. autoclass:: design_research_agents.model_selection.ModelSafetyConstraints

.. autoclass:: design_research_agents.model_selection.ModelSelectionConstraints

.. autoclass:: design_research_agents.model_selection.ModelSelectionDecision

.. autoclass:: design_research_agents.model_selection.ModelSelectionIntent

.. autoclass:: design_research_agents.model_selection.ModelSelectionPolicyConfig

.. autoclass:: design_research_agents.model_selection.ModelSelector
   :members:
   :undoc-members:

.. autoclass:: design_research_agents.model_selection.ModelSpec

Internal Modules
----------------

The underscored modules below are documented for contributor visibility. Public
usage should prefer ``design_research_agents.model_selection`` and the
top-level exports in ``design_research_agents``.

.. automodule:: design_research_agents._model_selection._catalog
   :no-index:

.. automodule:: design_research_agents._model_selection._hardware
   :no-index:

.. automodule:: design_research_agents._model_selection._policy
   :no-index:

.. automodule:: design_research_agents._model_selection._selector
   :no-index:

.. automodule:: design_research_agents._model_selection._types
   :no-index:
