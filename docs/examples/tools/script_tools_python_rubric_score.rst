Script Tools / Python / Rubric Score
====================================

Source: ``examples/tools/script_tools/python/rubric_score.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/tools/script_tools/python/rubric_score.py

Motivation
----------

Script tool header metadata.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Tool input"] --> B["Tool runtime"]
       B --> C["rubric score result"]
       C --> D["Artifacts and trace"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `tools` use-cases and run `rubric_score`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/tools/script_tools/python/rubric_score.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/tools/script_tools/python/rubric_score.py
   :language: python
   :linenos:
