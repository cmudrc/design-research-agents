Reinforcement Learning CartPole
===============================

Source: ``examples/patterns/reinforcement_learning_cartpole.py``

Introduction
------------

This example learns when to push left or right in Gymnasium's ``CartPole-v1``
inverted-pendulum environment. Unlike controller selection, every
reinforcement-learning action applies one force for one simulation timestep.
The built-in tabular Monte Carlo policy learns state-action values from complete
episodes without a preprogrammed balancing controller.

Technical Implementation
------------------------

From a source checkout, install the optional environment dependency with
``python -m pip install -e ".[rl]"``. The published-package equivalent is
``python -m pip install "design-research-agents[rl]"``. Then:

1. Reset CartPole with a different deterministic seed for each training episode.
2. Discretize pole angle and pole angular velocity into a compact state key.
3. Let ``ReinforcementLearningPattern`` choose ``push_left`` or ``push_right``
   at every simulation timestep.
4. Update tabular :math:`Q(s, a)` estimates from each complete episode return.
5. Evaluate the greedy learned policy on 50 held-out seeded initial states.

.. mermaid::

   flowchart LR
       A["Observe CartPole state"] --> B["Discretize angle and angular velocity"]
       B --> C["Select left or right force"]
       C --> D["Advance Gymnasium one timestep"]
       D --> E{"Failure or 200 steps?"}
       E -->|Continue| A
       E -->|Episode done| F["Update tabular state-action values"]
       F --> A

.. literalinclude:: ../../../examples/patterns/reinforcement_learning_cartpole.py
   :language: python
   :lines: 61-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/reinforcement_learning_cartpole.py

Output:

.. code-block:: text

   {
     "episodes_completed": 100,
     "evaluation_episodes": 50,
     "evaluation_mean_steps": 200.0,
     "evaluation_min_steps": 200,
     "first_20_mean_steps": 46.9,
     "last_20_mean_steps": 200.0,
     "learned_state_bins": 91,
     "success": true,
     "terminated_reason": "max_episodes_reached",
     "unseen_evaluation_states": 0,
     "value_mode": "state_action"
   }

References
----------

- `Gymnasium CartPole environment <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_
- `Gymnasium basic usage <https://gymnasium.farama.org/introduction/basic_usage/>`_
- `Sutton and Barto, Reinforcement Learning: An Introduction <http://incompleteideas.net/book/the-book-2nd.html>`_
