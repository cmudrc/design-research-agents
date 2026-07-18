Reinforcement Learning Custom Policy
====================================

Source: ``examples/patterns/reinforcement_learning_custom_policy.py``

Introduction
------------

Custom policies let the reinforcement learning pattern orchestrate state-dependent
or LLM-backed decisions without owning the learning algorithm. This example injects
an actor callable into a small feedback-guided policy. The deterministic actor keeps
the example reproducible; the same callable boundary can invoke an LLM that reads
the state and accumulated reward memory before selecting an action.

Technical Implementation
------------------------

1. Define a two-stage design environment where the best action changes by stage.
2. Implement the three-method structural policy contract: ``select_action``,
   ``update``, and ``get_params``.
3. Inject a deterministic actor that explores untried actions before exploiting the
   highest remembered reward; an LLM-backed actor can use the same inputs.
4. Run the public pattern and print the learned state-dependent strategy.

.. mermaid::

   flowchart LR
       A["Design stage state"] --> B["Injected actor"]
       B --> C["Custom policy selects action"]
       C --> D["Environment returns reward"]
       D --> E["Policy updates reward memory"]
       E --> B
       E --> F["ExecutionResult with traces"]

.. literalinclude:: ../../../examples/patterns/reinforcement_learning_custom_policy.py
   :language: python
   :lines: 54-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning_custom_policy.py

Output:

.. code-block:: text

   {
     "best_episode_reward": 5.0,
     "episodes_completed": 6,
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

- `Reflexion: Language Agents with Verbal Reinforcement Learning <https://arxiv.org/abs/2303.11366>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `Python typing protocols <https://typing.python.org/en/latest/spec/protocol.html>`_
