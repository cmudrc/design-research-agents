Local
=====

Source: ``examples/model_selection/local.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/model_selection/local.py

Motivation
----------

Run traced local-first model selection for a design summarization task.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Task profile"] --> B["Model selector"]
       B --> C["local decision"]
       C --> D["Decision payload and trace"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `model_selection` use-cases and run `local`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/model_selection/local.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/model_selection/local.py
   :language: python
   :linenos:
