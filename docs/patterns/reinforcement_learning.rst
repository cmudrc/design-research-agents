Reinforcement Learning Pattern
==============================

``ReinforcementLearningPattern`` runs a bounded episodic interaction loop
through the standard workflow and ``ExecutionResult`` contracts. The package
owns orchestration, reproducibility, and trace capture; the environment owns
transitions and rewards, while the policy owns learning.

The pattern has no machine-learning dependency. It supports three deliberately
small policy paths:

- ``actions`` estimates one global value per discrete action, suitable for
  bandits such as agent, prompt, tool, or model selection;
- ``actions`` plus ``state_key`` estimates tabular Monte Carlo state-action
  values for small discrete state spaces; and
- ``policy`` accepts any object implementing the structural custom-policy
  contract.

Global Action Values
--------------------

This one-step benchmark learns which agent strategy receives the highest
measured reward:

.. code-block:: python

   from collections.abc import Mapping

   from design_research_agents import ReinforcementLearningPattern


   rewards = {"direct": 0.54, "critique": 0.88, "tool_assisted": 0.73}


   def reset() -> dict[str, object]:
       return {"benchmark": "field_service_planning"}


   def step(
       state: Mapping[str, object],
       action: str | Mapping[str, object],
   ) -> tuple[dict[str, object], float, bool]:
       if not isinstance(action, str):
           raise TypeError("This benchmark requires a string action.")
       return dict(state), rewards[action], True


   pattern = ReinforcementLearningPattern(
       environment_reset=reset,
       environment_step=step,
       actions=list(rewards),
       max_episodes=60,
       max_steps_per_episode=1,
       random_seed=7,
   )
   result = pattern.run("Learn which agent strategy performs best.")

   print(result.output["final_output"]["final_policy_params"]["action_values"])

See :doc:`../examples/patterns/reinforcement_learning` for the complete runnable
example and recorded output.

Simulation-Backed State-Action Learning
---------------------------------------

The :doc:`../examples/patterns/reinforcement_learning_cartpole` example applies
one learned left or right force per Gymnasium ``CartPole-v1`` timestep. It
discretizes pole angle and angular velocity with ``state_key`` and estimates
tabular Monte Carlo :math:`Q(s, a)` from complete episode returns. Install the
optional environment dependency with
``python -m pip install "design-research-agents[rl]"``.

The example deliberately omits a preprogrammed feedback controller. Its
evaluation reports learning progress, held-out balance duration, table size, and
whether evaluation encountered any state bins absent from training.

Tabular State-Action Values
---------------------------

Pass ``state_key`` when the best action depends on a small discrete state. The
callable must return a stable non-empty string. The built-in policy then records
discounted Monte Carlo returns for each state-action pair:

.. code-block:: python

   pattern = ReinforcementLearningPattern(
       environment_reset=reset,
       environment_step=step,
       actions=["explore", "refine"],
       state_key=lambda state: str(state["stage"]),
       max_episodes=100,
       random_seed=7,
   )

   result = pattern.run("Learn a stage-dependent design strategy.")
   q_values = result.output["final_output"]["final_policy_params"]["q_values"]

This tabular mode estimates :math:`Q(s, a)` from complete episode returns. It is
not one-step Q-learning and does not approximate values for unseen or continuous
states. Use a custom policy for those settings.

Custom Policies And LLM Actors
------------------------------

For function approximation, continuous actions, external RL libraries, or
LLM-backed action selection, provide ``policy`` and omit ``actions`` and
``state_key``. The object is structural; it does not need to inherit from a
package base class:

.. code-block:: python

   class Policy:
       def select_action(self, state):
           ...

       def update(self, trajectory):
           return {"loss": 0.0}

       def get_params(self):
           return {"checkpoint": "episode-latest"}

``select_action`` may return a string or mapping. ``update`` receives one
episode as ``(state, action, reward)`` tuples. ``update`` statistics and
``get_params`` snapshots should be JSON-safe; trace capture also applies a
bounded best-effort normalization so logging cannot alter execution values.

The :doc:`../examples/patterns/reinforcement_learning_custom_policy` example
injects an actor callable into a feedback-guided policy. Its deterministic actor
can be replaced by a callable that invokes an LLM with the current state and
accumulated reward memory.

Stopping Semantics
------------------

Training stops at ``max_episodes`` by default. The pattern does not claim that a
generic reward sequence proves policy convergence.

For bounded experiments where a stable reward signal is itself a useful early
stop, set ``convergence_threshold`` and ``convergence_episodes`` explicitly.
This activates a private reward-stability heuristic and reports
``terminated_reason="reward_stable"``. It does not report policy convergence.

Results And Traces
------------------

The canonical result includes:

- ``final_output``: best reward and episode, episode count, reward history, and
  final policy parameters;
- ``details``: configuration, the initial policy snapshot, and per-episode
  transition traces with one post-update policy snapshot each;
- ``metadata``: mode, value mode, request id, dependency keys, reproducibility
  settings, and output-contract version; and
- ``workflow``: the underlying loop execution payload.

Trace states, actions, policy statistics, and policy parameters are recursively
normalized into bounded JSON-safe snapshots. The policy trajectory receives
defensive copies of states and actions so later in-place environment mutation
cannot rewrite prior experience.

Full transition traces still grow with episodes and steps. Keep bounds deliberate
when states or policy snapshots are large.

Reproducibility And Reuse
-------------------------

Set ``random_seed`` when using either built-in value mode. Environment callables,
``state_key``, and custom policies remain responsible for their own deterministic
behavior.

The policy object is updated in place. Reusing one pattern instance continues
from its learned policy; construct a new pattern for an independent replicate.

Reference
---------

- `Sutton and Barto, Reinforcement Learning: An Introduction
  <http://incompleteideas.net/book/the-book-2nd.html>`_
- `Reflexion: Language Agents with Verbal Reinforcement Learning
  <https://arxiv.org/abs/2303.11366>`_
- `Python typing protocols <https://typing.python.org/en/latest/spec/protocol.html>`_
