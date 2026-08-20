Seeded Random Baseline Agent
============================

Source: ``examples/agents/seeded_random_baseline_agent.py``

Introduction
------------

This example treats a seeded random baseline as a first-class study participant for packaged-problem-style
experiments. It keeps the setup dependency-light by using a local decision-problem stub that mirrors the
public candidate-iteration contract instead of importing sibling repositories, while still using the same
``run(prompt, dependencies=...)`` contract as the other public agents.

Technical Implementation
------------------------

1. Define a small packaged-problem-style decision stub with ``iter_candidates()`` and ``evaluate()``.
2. Run ``SeededRandomBaselineAgent`` with a fixed seed and pass the packaged problem through
   ``dependencies`` so the sampled control candidate is reproducible.
3. Compare the random control condition against a deterministic greedy baseline that always picks the
   highest-scoring candidate.
4. Print JSON that could be dropped into lightweight experiment wiring or docs examples.

.. mermaid::

   flowchart LR
       A["Local decision problem stub"] --> B["SeededRandomBaselineAgent(seed=7)"]
       A --> C["Greedy comparator"]
       B --> D["Random control candidate"]
       C --> E["Deterministic candidate"]
       D --> F["JSON comparison output"]
       E --> F

.. literalinclude:: ../../../examples/agents/seeded_random_baseline_agent.py
   :language: python
   :lines: 51-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/agents/seeded_random_baseline_agent.py

Example output shape:

.. code-block:: text

   {
     "problem_id": "local_heat_sink_layout",
     "random_condition": {
       "candidate": {"fin_count": 6.0, "gap_mm": 2.0, "wall_mm": 1.0},
       "score": 0.89
     },
     "greedy_condition": {
       "candidate": {"fin_count": 8.0, "gap_mm": 2.0, "wall_mm": 1.5},
       "score": 1.065
     }
   }

References
----------

- `Python random module <https://docs.python.org/3/library/random.html>`_
- `HELM: Holistic Evaluation of Language Models <https://arxiv.org/abs/2211.09110>`_
- `Design Research Agents documentation <https://cmudrc.github.io/design-research-agents/>`_
