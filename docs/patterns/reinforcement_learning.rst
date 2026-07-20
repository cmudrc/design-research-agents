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

The pattern owns one training run and optional frozen-policy evaluation. Study
replicates, condition scheduling, checkpoints, and statistical comparisons stay
with the sibling experiments and analysis libraries.

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
whether evaluation encountered any state bins absent from training. It calls
``pattern.evaluate(...)`` rather than reading the Q-table to execute the policy.

Tabular State-Action Values
---------------------------

Pass ``state_key`` when the best action depends on a small discrete state. The
callable must return a stable non-empty string. The built-in policy then records
discounted first-visit Monte Carlo returns for each state-action pair:

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

This tabular mode estimates :math:`Q(s, a)` with first-visit Monte Carlo over
complete episode returns: when the same state-action pair recurs within one
episode, only the return following its first occurrence updates the estimate. It
is not one-step Q-learning and does not approximate values for unseen or
continuous states. Use a custom policy for those settings.

Environment Transitions
-----------------------

Existing environments can continue returning the compact three-value result:

.. code-block:: python

   next_state, reward, done

New environments should prefer the Gymnasium-style five-value result:

.. code-block:: python

   next_state, reward, terminated, truncated, info

``terminated`` means the environment reached an MDP terminal state.
``truncated`` means an external limit, such as a time or step budget, ended the
episode. The pattern also marks its own ``max_steps_per_episode`` limit as a
truncation. Both forms produce a normalized ``RLTransition`` containing the
pre-action state, action, reward, next state, terminal flags, and auxiliary
``info`` mapping.

Custom Policies And LLM Actors
------------------------------

For function approximation, continuous actions, external RL libraries, or
agent-backed action selection, provide ``policy`` and omit ``actions`` and
``state_key``. The object is structural; it does not need to inherit from a
package base class. Existing episode-updated policies remain valid:

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
``get_params`` output should be JSON-safe. For a large model, return compact
metadata or a checkpoint reference rather than model weights.

A policy that needs immediate transition updates can replace ``update`` with
the following paired hooks:

.. code-block:: python

   class OnlinePolicy:
       def select_action(self, state):
           ...

       def observe_transition(self, transition):
           # Learn from one RLTransition immediately.
           ...

       def end_episode(self, transitions):
           return {"loss": 0.0}

       def get_params(self):
           return {"checkpoint": "episode-latest"}

``observe_transition`` and ``end_episode`` must be implemented together. This
small lifecycle supports online custom policies without introducing a trainer,
replay buffer, or tensor framework into this package.

The :doc:`../examples/patterns/reinforcement_learning_custom_policy` example
uses ``DirectLLMCall`` as the policy actor. The agent receives the current design
stage, available actions, and accumulated reward memory, then returns one JSON
action. A deterministic local client keeps the example offline; another LLM
client can be substituted without changing the RL orchestration.

Evaluation And Replicated Studies
---------------------------------

``pattern.evaluate(...)`` executes the current policy without updating it. The
built-in policy uses deterministic greedy actions and does not advance the
training RNG, so repeated evaluations of a frozen policy are reproducible.
Deterministic evaluation resolves two cases by the order actions were passed to
the constructor: a tie for the best value selects the first such action, and a
state never seen during training falls back to the first configured action.
Training-time selection instead breaks ties uniformly at random. A custom policy
must provide ``select_evaluation_action(state)`` or the caller must pass
``action_selector``. Evaluation can use separate reset and step callables for
held-out environments:

.. code-block:: python

   evaluation = pattern.evaluate(
       episodes=50,
       environment_reset=held_out_reset,
       environment_step=environment_step,
   )
   mean_reward = evaluation.output["final_output"]["mean_reward"]

Do not add another replicate loop around the pattern for a research study.
``design-research-experiments`` already owns deterministic seeds, independent
replicates, parallel execution, resume, and canonical artifacts. Use one study
run to construct, train, and evaluate one fresh policy:

.. code-block:: python

   import design_research_experiments as drex


   def run_condition(run_spec, condition):
       pattern = build_pattern(seed=run_spec.seed, condition=condition)
       training = pattern.run("Train the policy.")
       evaluation = pattern.evaluate(episodes=20)
       return drex.RunOutput(
           outputs={"training": training.summary()},
           metrics={
               "primary_outcome": evaluation.output["final_output"]["mean_reward"],
           },
       )

   results = drex.run_study(study, condition_runner=run_condition)

Exported run and evaluation tables can then be passed to
``design-research-analysis`` for bootstrap intervals, effects, and condition
comparisons.

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
- ``details``: configuration, the initial policy snapshot, and configurable
  per-episode traces;
- ``metadata``: mode, value mode, request id, dependency keys, reproducibility
  settings, and output-contract version; and
- ``workflow``: the underlying loop execution payload.

``trace_detail="summary"`` retains episode metrics only.
``trace_detail="transitions"`` is the default and adds normalized step
transitions. ``trace_detail="full"`` also stores one policy snapshot after every
episode. The final policy state is always retained regardless of trace detail.

Trace states, actions, ``info``, policy statistics, and optional policy
parameters are recursively normalized into bounded JSON-safe snapshots. Policy
experience receives defensive copies so later in-place environment mutation
cannot rewrite prior experience.

Transition traces still grow with episodes and steps. Prefer ``summary`` for
large studies and ``full`` only while debugging small policies.

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
- `Agent Lightning: Train ANY AI Agents with Reinforcement Learning
  <https://arxiv.org/abs/2508.03680>`_
- `Gymnasium environment API <https://gymnasium.farama.org/api/env/>`_
- `Python typing protocols <https://typing.python.org/en/latest/spec/protocol.html>`_
