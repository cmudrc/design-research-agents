Simulated Annealing
===================

Source: ``examples/patterns/simulated_annealing.py``

Introduction
------------

Simulated annealing is useful when a design space has a numeric objective,
lightweight constraints, and local moves that may temporarily get worse before
finding a better basin. This example keeps the delegates deterministic so the
runtime contract is easy to inspect without an LLM dependency.

Technical Implementation
------------------------

1. Define a local objective delegate for a one-dimensional quadratic target.
2. Define a neighbor delegate that proposes bounded local moves.
3. Execute ``SimulatedAnnealingPattern.run(...)`` through the public patterns API.
4. Print a compact JSON payload for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Initial state"] --> B["SimulatedAnnealingPattern.run(...)"]
       B --> C["neighbor_delegate proposes local moves"]
       C --> D["objective_delegate scores each state"]
       D --> E["Metropolis acceptance + convergence checks"]
       E --> F["ExecutionResult/payload"]
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/simulated_annealing.py
   :language: python
   :lines: 49-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/simulated_annealing.py

Example output shape:

.. code-block:: text

   {
     "best_objective_value": 0.0,
     "best_state": {
       "x": 3.0
     },
     "iterations": 6,
     "success": true,
     "terminated_reason": "max_iterations_reached"
   }

References
----------

- `Optimization by Simulated Annealing <https://www.science.org/doi/10.1126/science.220.4598.671>`_
- `Equation of State Calculations by Fast Computing Machines <https://doi.org/10.1063/1.1699114>`_
- `Stochastic Relaxation, Gibbs Distributions, and Bayesian Image Restoration <https://doi.org/10.1109/TPAMI.1984.4767596>`_
