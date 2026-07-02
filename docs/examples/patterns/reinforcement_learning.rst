Reinforcement Learning
======================

Source: ``examples/patterns/reinforcement_learning.py``

Introduction
------------

The reinforcement learning pattern runs an episodic agent-environment loop where
a policy learns from cumulative rewards. This example uses a simple grid navigation
environment with discrete actions. The agent starts at position 0 and learns to
move right to reach the goal at position 3. All delegates are deterministic so the
runtime contract is easy to inspect without an LLM dependency.

Technical Implementation
------------------------

1. Define an environment reset delegate that returns the initial state.
2. Define an environment step delegate that applies an action and returns the next
   state, reward, and done flag.
3. Execute ``ReinforcementLearningPattern.run(...)`` through the public patterns API.
4. Print a compact JSON payload showing the learned policy and reward history.

.. mermaid::

   flowchart LR
       A["Initial state"] --> B["ReinforcementLearningPattern.run(...)"]
       B --> C["environment_reset starts each episode"]
       C --> D["EpsilonGreedyPolicy selects action"]
       D --> E["environment_step returns next_state, reward, done"]
       E --> F["Trajectory collected until done"]
       F --> G["Monte Carlo policy update"]
       G --> H["ExecutionResult/payload"]
       H --> I["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/reinforcement_learning.py
   :language: python
   :lines: 51-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning.py

Example output shape:

.. code-block:: text

{
    "best_episode_reward": 7.0,
    "best_episode_index": ...,
    "episodes_completed": ...,
    "success": true,
    "terminated_reason": "converged"
   }

References
----------

- `Reinforcement Learning <https://en.wikipedia.org/wiki/Reinforcement_learning>`_
- `Q-Learning <https://en.wikipedia.org/wiki/Q-learning>`_
- `Epsilon-greedy Algorithm <https://en.wikipedia.org/wiki/Epsilon-greedy_algorithm>`_
