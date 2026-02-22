Tracing Modules
===============

This page lists internal tracing modules. These underscored module paths are
documented for contributor visibility and are not API-stable.

Observed Event Additions
------------------------

Recent tracing additions emit payload-observation events at key boundaries:

- ``ModelRequestObserved``
- ``ModelResponseObserved``
- ``ToolInvocationObserved``
- ``ToolResultObserved``
- ``WorkflowStepContextObserved``
- ``WorkflowStepResultObserved``

These events are additive and do not replace existing run/model/tool span events.

Payload Redaction and Preview Policy
------------------------------------

Observed payload events apply a safe-default policy before writing:

- Sensitive keys are recursively masked (for example ``api_key``, ``token``,
  ``authorization``, ``password``, ``secret``, ``access_token``,
  ``refresh_token``).
- Long string fields are truncated to bounded previews for practical trace size.

.. automodule:: design_research_agents._tracing._config
   :members:
   :undoc-members:
   :no-index:

.. automodule:: design_research_agents._tracing._context
   :members:
   :undoc-members:
   :no-index:

.. automodule:: design_research_agents._tracing._emitters
   :members:
   :undoc-members:
   :no-index:

.. automodule:: design_research_agents._tracing._session
   :members:
   :undoc-members:
   :no-index:

.. automodule:: design_research_agents._tracing._sinks
   :members:
   :undoc-members:
   :no-index:

.. automodule:: design_research_agents._tracing._utils
   :members:
   :undoc-members:
   :no-index:

.. automodule:: design_research_agents._tracing._analysis
   :members:
   :undoc-members:
   :no-index:
