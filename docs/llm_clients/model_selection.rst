Model Selection
===============

Use model selection to keep client/model choices consistent under explicit
constraints.

Core API
--------

- ``ModelSelector``: flat selector interface for task/constraint inputs.
- ``ModelSelector.select(..., output="decision")`` returns internal
  ``ModelSelectionDecision`` metadata.
- ``ModelSelector.select(..., output="client_config")`` returns a plain mapping
  with ``provider``, ``model_id``, ``client_class``, and constructor ``kwargs``.
- ``ModelSelector.select(..., output="client")`` (default) returns an
  instantiated LLM client.

Underlying policy/types
-----------------------

``ModelSelector`` delegates to internal policy components:

- ``ModelSelectionPolicy`` for candidate scoring and decisioning
- ``ModelSelectionIntent`` for task/priority intent
- ``ModelSelectionConstraints`` for local/provider/cost/latency bounds
- ``HardwareProfile`` for hardware-aware local fit checks

Constraint examples
-------------------

Local-only selection:

.. code-block:: python

   from design_research_agents import ModelSelector

   selector = ModelSelector()
   decision = selector.select(
       task="summarize interview notes",
       priority="balanced",
       require_local=True,
       output="decision",
   )

Cost-capped selection:

.. code-block:: python

   selector = ModelSelector()
   decision = selector.select(task="summarize", max_cost_usd=0.01, output="decision")

Latency-capped selection:

.. code-block:: python

   selector = ModelSelector()
   decision = selector.select(task="chat", max_latency_ms=1200, output="decision")

Local model fit behavior
------------------------

``ModelSelector`` evaluates local candidates against available RAM/VRAM hints
and can prefer remote candidates under high system load. This preserves a
local-first posture while avoiding local models likely to overload the current
machine.

See examples
------------

- ``examples/model_selection/local.py``
- ``examples/model_selection/remote.py``
- ``examples/model_selection/README.md``

Related
-------

- :doc:`index`
