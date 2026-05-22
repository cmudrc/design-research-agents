Reinforcement Learning Pattern Starter
======================================

This page is a starter scaffold for
`issue #47 <https://github.com/cmudrc/design-research-agents/issues/47>`_.
It is intentionally narrower than a finished user guide: the goal is to give
the first implementation pass a shared contract, not to document an API that
already exists.

Implementation Target
---------------------

The first reinforcement-learning pattern should support a classic
agent-environment loop for sequential design tasks:

#. Reset an environment to an initial design state.
#. Ask a policy for an action.
#. Apply the action through the environment.
#. Observe the next state, reward, termination flag, and metadata.
#. Update the policy from the transition.
#. Repeat for the requested number of episodes or until the environment stops.

Keep the first implementation deterministic by default and dependency-light.
The initial algorithm can be a simple actor-critic or tabular policy update
as long as the update is visible in tests. PPO or other larger methods can be
added once the base loop is stable.

Suggested Public Shape
----------------------

The public facade should eventually expose ``RLAgentPattern`` from
``design_research_agents.patterns`` once the implementation and tests are
ready. Keep implementation details behind the same private pattern boundary
used by the existing pattern facades.

The core contracts should stay small:

.. code-block:: python

   State = Mapping[str, object]
   Action = Mapping[str, object]
   Reward = float

   class Environment(Protocol):
       def reset(self) -> State: ...
       def step(self, action: Action) -> tuple[State, Reward, bool, Mapping[str, object]]: ...

   class Policy(Protocol):
       def select_action(self, state: State) -> Action: ...
       def update(self, transition: Transition) -> Mapping[str, object]: ...

``Transition`` should capture at least state, action, reward, next state,
done, episode, and step index. Any policy update metadata should be JSON-safe
so it can flow into result details and trace output.

Relationship To Multi-Armed Bandits
-----------------------------------

The multi-armed bandit issue should be able to reuse this foundation. Treat a
bandit as a one-step environment with a smaller action space and no long-term
state transition. That keeps the RL agent pattern as the more general base and
lets a future bandit pattern subclass or compose the same contracts.

Result And Trace Expectations
-----------------------------

The ``ExecutionResult`` payload should follow the existing pattern result
shape used by ``SimulatedAnnealingPattern`` and related patterns:

- ``final_output`` should include the best or final policy summary, total
  episodes, total steps, cumulative reward, and termination reason.
- ``details`` should include episode summaries plus a compact transition trace
  with states, actions, rewards, done flags, and policy update metadata.
- ``metadata`` should include the pattern mode, request id, dependency keys,
  and configuration values that affect reproducibility.

Validation Plan
---------------

The first implementation PR should include focused tests for:

- input validation for episode count, max steps, environment, and policy;
- deterministic interaction over ``N`` episodes with a dummy design
  simulation environment;
- a simple converging policy update where the selected action improves after
  rewards are observed;
- trace details that include states, actions, rewards, and policy update
  metadata;
- public API exports if ``RLAgentPattern`` is exposed from
  ``design_research_agents.patterns`` or the top-level package.

Reference
---------

- `Hugging Face Deep RL Course, Unit 0 <https://huggingface.co/learn/deep-rl-course/unit0/introduction>`_
