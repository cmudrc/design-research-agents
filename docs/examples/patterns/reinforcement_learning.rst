Reinforcement Learning
======================

Source: ``examples/patterns/reinforcement_learning.py``

Introduction
------------

The built-in reinforcement learning policy is an honest global action-value
baseline. This example repeatedly selects one of three agent strategies for the
same benchmark family and learns which strategy receives the highest measured
reward. The seeded policy and deterministic benchmark make the result reproducible
without an LLM or machine-learning dependency.

Technical Implementation
------------------------

1. Define an environment reset delegate that identifies the benchmark family.
2. Define a one-step environment delegate that scores each selected agent strategy.
3. Execute ``ReinforcementLearningPattern.run(...)`` with discrete ``actions`` and
   no ``state_key``, selecting global-action value mode.
4. Print the learned action values and bounded training summary.

.. mermaid::

   flowchart LR
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
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning.py

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

References
----------

- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
- `Multi-armed bandit algorithms and empirical evaluation <https://arxiv.org/abs/1003.0146>`_
- `AgentBench: Evaluating LLMs as Agents <https://arxiv.org/abs/2308.03688>`_
