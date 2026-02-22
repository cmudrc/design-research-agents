Propose Critic
==============

Source: ``examples/patterns/propose_critic.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/propose_critic.py

Motivation
----------

Run traced ``ReflexionPattern`` for iterative design-summary refinement.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Pattern prompt"] --> B["Pattern orchestration"]
       B --> C["propose critic result"]
       C --> D["Trace metadata"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `patterns` use-cases and run `propose_critic`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/patterns/propose_critic.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/patterns/propose_critic.py
   :language: python
   :linenos:
