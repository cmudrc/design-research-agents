Reinforcement Learning Pattern
==============================

``ReinforcementLearningPattern`` runs a bounded episodic interaction loop
through the standard workflow and ``ExecutionResult`` contracts. It is useful
when a design process can be expressed as four small pieces:

#. an environment reset callable;
#. an environment transition callable;
#. a policy that selects actions and learns from complete episodes; and
#. a scalar reward that makes improvement observable.

The pattern has no machine-learning dependency. Passing discrete ``actions``
constructs a seeded epsilon-greedy Monte Carlo baseline. Passing ``policy``
instead uses any object that implements the structural policy contract described
below.

Quick Start
-----------

This one-step environment learns which of two design choices is rewarded:

.. code-block:: python

   from collections.abc import Mapping

   from design_research_agents import ReinforcementLearningPattern


   def reset() -> dict[str, object]:
       return {"trial": 0}


   def step(
       state: Mapping[str, object],
       action: str | Mapping[str, object],
   ) -> tuple[dict[str, object], float, bool]:
       del state
       if not isinstance(action, str):
           raise TypeError("This environment requires a string action.")
       reward = 1.0 if action == "lightweight" else -1.0
       return {"trial": 1}, reward, True


   pattern = ReinforcementLearningPattern(
       environment_reset=reset,
       environment_step=step,
       actions=["lightweight", "rigid"],
       max_episodes=50,
       max_steps_per_episode=1,
       random_seed=7,
   )
   result = pattern.run("Learn which concept receives the higher reward.")

   print(result.success)
   print(result.output["final_output"]["final_policy_params"])

Default Policy Scope
--------------------

The default epsilon-greedy policy estimates one value per action and ignores
state. This makes it a transparent baseline for bandits and small environments
where one action has a consistent global value. It is not a general Q-learning,
actor-critic, or policy-gradient implementation.

For a state-dependent design problem, provide a custom ``policy`` and omit
``actions``. The object is structural; it does not need to inherit from a
package base class:

.. code-block:: python

   class Policy:
       def select_action(self, state):
           ...

       def update(self, trajectory):
           return {"loss": 0.0}

       def get_params(self):
           return {"checkpoint": "episode-latest"}

``select_action`` may return a string or a mapping. ``update`` receives one
episode as ``(state, action, reward)`` tuples. ``update`` statistics and
``get_params`` snapshots must be JSON-safe because they are included in traces
and result metadata.

Results And Traces
------------------

The canonical result includes:

- ``final_output``: best reward and episode, episode count, reward history, and
  final policy parameters;
- ``details``: configuration, per-episode transition traces, update statistics,
  and policy-parameter history;
- ``metadata``: mode, request id, dependency keys, reproducibility settings,
  and output-contract version; and
- ``workflow``: the underlying loop execution payload.

The full transition and parameter histories make short research runs auditable,
but they also grow with episodes and steps. Keep bounds deliberate when states
or policy snapshots are large.

Reproducibility And Reuse
-------------------------

Set ``random_seed`` when using the default policy. Environment callables and
custom policies remain responsible for their own random-number state.

The policy object is updated in place. Reusing one pattern instance continues
from its learned policy; construct a new pattern for an independent replicate.

Reference
---------

- `Sutton and Barto, Reinforcement Learning: An Introduction
  <http://incompleteideas.net/book/the-book-2nd.html>`_
- `Gymnasium environment API <https://gymnasium.farama.org/api/env/>`_
- `Gymnasium basic usage <https://gymnasium.farama.org/introduction/basic_usage/>`_
