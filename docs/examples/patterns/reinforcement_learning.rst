Reinforcement Learning
======================

Source: ``examples/patterns/reinforcement_learning.py``

Introduction
------------

<<<<<<< HEAD
The reinforcement learning pattern runs an episodic agent-environment loop where
a policy learns from cumulative rewards. This example uses a simple grid navigation
environment with discrete actions. The agent starts at position 0 and learns to
move right to reach the goal at position 3. All delegates are deterministic so the
runtime contract is easy to inspect without an LLM dependency.
=======
The built-in reinforcement learning policy is an honest global action-value
baseline. This example repeatedly selects one of three agent strategies for the
same benchmark family and learns which strategy receives the highest measured
reward. The seeded policy and deterministic benchmark make the result reproducible
without an LLM or machine-learning dependency.
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477

Technical Implementation
------------------------

<<<<<<< HEAD
1. Define an environment reset delegate that returns the initial state.
2. Define an environment step delegate that applies an action and returns the next
   state, reward, and done flag.
3. Execute ``ReinforcementLearningPattern.run(...)`` through the public patterns API.
4. Print a compact JSON payload showing the learned policy and reward history.
=======
1. Define an environment reset delegate that identifies the benchmark family.
2. Define a one-step environment delegate that scores each selected agent strategy.
3. Execute ``ReinforcementLearningPattern.run(...)`` with discrete ``actions`` and
   no ``state_key``, selecting global-action value mode.
4. Print the learned action values and bounded training summary.
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477

.. mermaid::

   flowchart LR
<<<<<<< HEAD
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
=======
       A["Benchmark task"] --> B["ReinforcementLearningPattern.run(...)"]
       B --> C["Epsilon-greedy strategy selection"]
       C --> D["Benchmark returns reward"]
       D --> E["Monte Carlo action-value update"]
       E --> C
       E --> F["ExecutionResult with traces"]
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/reinforcement_learning.py
   :language: python
   :lines: 55-
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning.py

<<<<<<< HEAD
Example output shape:

.. code-block:: text

{
    "best_episode_reward": 7.0,
    "best_episode_index": ...,
    "episodes_completed": ...,
    "success": true,
    "terminated_reason": "converged"
    }
=======
Output:

.. code-block:: text

   {
     "action_values": {
       "critique_then_answer": 0.88,
       "direct_answer": 0.54,
       "tool_assisted": 0.73
     },
     "best_episode_reward": 0.88,
     "episodes_completed": 60,
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "value_mode": "global_action"
   }
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477

References
----------

<<<<<<< HEAD
- `Reinforcement Learning <https://huggingface.co/learn/deep-rl-course/en/unit1/what-is-rl>`_
- `Q-Learning <https://huggingface.co/learn/deep-rl-course/en/unit2/q-learning>`_
- `Epsilon-greedy Algorithm <https://www.geeksforgeeks.org/machine-learning/epsilon-greedy-algorithm-in-reinforcement-learning/>`_
=======
- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
- `Multi-armed bandit algorithms and empirical evaluation <https://arxiv.org/abs/1003.0146>`_
- `AgentBench: Evaluating LLMs as Agents <https://arxiv.org/abs/2308.03688>`_
>>>>>>> 77df08ad501aebf3994ba244d33bfff09fcd7477
