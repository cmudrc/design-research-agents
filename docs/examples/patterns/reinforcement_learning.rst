Reinforcement Learning
======================

Source: ``examples/patterns/reinforcement_learning.py``

Introduction
------------

The reinforcement learning pattern runs an episodic agent-environment loop where
a policy learns from cumulative rewards. This example uses a simple grid navigation
environment with discrete actions. The agent starts at position 0 and learns to
move right to reach the goal at position 3. The seeded policy and deterministic
environment make the result reproducible without an LLM dependency.

Technical Implementation
------------------------

1. Define an environment reset delegate that returns the initial state.
2. Define an environment step delegate that applies an action and returns the next
   state, reward, and done flag.
3. Execute ``ReinforcementLearningPattern.run(...)`` through the public top-level API.
4. Print a compact JSON payload showing the training summary and learned action values.

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
   :lines: 56-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning.py

Output:

.. code-block:: text

   {
     "best_episode_index": 9,
     "best_episode_reward": 8.0,
     "episodes_completed": 57,
     "final_q_values": {
       "left": -1.7394775438422179,
       "right": 6.643450112696429,
       "stay": 1.0547376105811548
     },
     "success": true,
     "terminated_reason": "converged"
   }

References
----------

- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
- `Gymnasium environment API <https://gymnasium.farama.org/api/env/>`_
- `Gymnasium basic usage <https://gymnasium.farama.org/introduction/basic_usage/>`_
