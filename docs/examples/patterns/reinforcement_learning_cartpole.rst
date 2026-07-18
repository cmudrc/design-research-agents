Reinforcement Learning CartPole
===============================

Source: ``examples/patterns/reinforcement_learning_cartpole.py``

Introduction
------------

This example learns which feedback-controller design balances Gymnasium's
``CartPole-v1`` inverted pendulum most reliably. Each reinforcement-learning
action selects one complete controller, and the environment returns the number
of balanced simulation steps as reward. The formulation is a transparent
controller-selection problem, not per-timestep Q-learning.

Technical Implementation
------------------------

From a source checkout, install the optional environment dependency with
``python -m pip install -e ".[rl]"``. The published-package equivalent is
``python -m pip install "design-research-agents[rl]"``. Then:

1. Reset CartPole with a different deterministic seed for each training episode.
2. Let ``ReinforcementLearningPattern`` select one candidate feedback controller.
3. Apply that controller's left/right force decisions for at most 200 simulation
   steps as one macro action.
4. Learn global action values from the number of balanced steps.
5. Evaluate the best learned controller on 25 held-out seeded initial states.

.. mermaid::

   flowchart LR
       A["Seeded CartPole reset"] --> B["Select feedback controller"]
       B --> C["Apply left or right force"]
       C --> D{"Failure or 200 steps?"}
       D -->|Continue| C
       D -->|Done| E["Reward equals balanced steps"]
       E --> F["Update controller value"]
       F --> B

.. literalinclude:: ../../../examples/patterns/reinforcement_learning_cartpole.py
   :language: python
   :lines: 63-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning_cartpole.py

Output:

.. code-block:: text

   {
     "action_values": {
       "angle_only": 48.6,
       "full_state": 149.8,
       "overcorrected": 189.29,
       "rate_assisted": 200.0
     },
     "episodes_completed": 40,
     "evaluation_mean_steps": 200.0,
     "evaluation_min_steps": 200,
     "learned_controller": "rate_assisted",
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "value_mode": "global_action"
   }

References
----------

- `Gymnasium CartPole environment <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_
- `Gymnasium basic usage <https://gymnasium.farama.org/introduction/basic_usage/>`_
- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
