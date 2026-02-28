Networked and Blackboard Patterns
=================================

``RoundBasedCoordinationPattern`` and ``BlackboardPattern`` provide peer-only coordination
without a central orchestrator agent.

Core behavior
-------------

- Round-based execution with deterministic peer order (sorted peer ids).
- No manager/supervisor role required.
- Explicit termination reasons:

  - ``converged``
  - ``max_rounds_reached``
  - ``explicit_stop``
  - ``peer_failure``

Blackboard specialization
-------------------------

``BlackboardPattern`` keeps an explicit shared board payload:

.. code-block:: python

   {
       "task": str,
       "round": int,
       "messages": list,
       "proposals": dict,
       "decisions": dict,
       "history": list,
   }

The reducer merges per-peer contributions by channel, appends immutable message
history entries, computes ``state_hash``, and can declare convergence when the
hash stays unchanged for configured stability rounds.

Background references
---------------------

- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `CAMEL <https://arxiv.org/abs/2303.17760>`_
- `MetaGPT <https://arxiv.org/abs/2308.00352>`_
- `ChatDev <https://arxiv.org/abs/2307.07924>`_

These are examples of LLM multi-agent coordination; these patterns are
framework-native and intentionally minimal.

Example
-------

See ``examples/patterns/coordination_patterns.py``.
