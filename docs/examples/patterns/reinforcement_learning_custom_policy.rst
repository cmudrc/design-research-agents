Reinforcement Learning Custom Policy
====================================

Source: ``examples/patterns/reinforcement_learning_custom_policy.py``

Introduction
------------

An existing agent or workflow can act as a reinforcement-learning policy without
creating another orchestration pattern. This example places ``DirectLLMCall``
behind a small structural policy: the agent sees the current design stage,
available actions, and accumulated reward feedback, then selects one structured
action. A deterministic local client keeps the example offline and reproducible;
the same ``DirectLLMCall`` boundary can use any configured LLM client.

Technical Implementation
------------------------

1. Define a two-stage design environment where the best action changes by stage.
2. Wrap ``DirectLLMCall`` in a policy that prompts for one JSON action.
3. Learn from rich ``RLTransition`` objects through ``observe_transition`` and
   close each episode through ``end_episode``.
4. Train for six episodes, then evaluate the learned policy without updating it.

.. mermaid::

   flowchart LR
       A["Design stage + reward memory"] --> B["DirectLLMCall policy actor"]
       B --> C["Structured explore or refine action"]
       C --> D["Environment transition"]
       D --> E["Policy observes RLTransition"]
       E --> A
       E --> F["Frozen evaluation"]

.. literalinclude:: ../../../examples/patterns/reinforcement_learning_custom_policy.py
   :language: python
   :lines: 56-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning_custom_policy.py

Output:

.. code-block:: text

   {
     "agent_calls": 18,
     "best_episode_reward": 5.0,
     "episodes_completed": 6,
     "evaluation_mean_reward": 5.0,
     "learned_actions": {
       "concept": "explore",
       "detail": "refine"
     },
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "value_mode": "custom"
   }

References
----------

- `Agent Lightning: Train ANY AI Agents with Reinforcement Learning <https://arxiv.org/abs/2508.03680>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Python typing protocols <https://typing.python.org/en/latest/spec/protocol.html>`_
