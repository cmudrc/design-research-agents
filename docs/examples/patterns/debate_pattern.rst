Debate Pattern
==============

Source: ``examples/patterns/debate_pattern.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/debate_pattern.py

Motivation
----------

Run traced ``DebatePattern`` on an engineering design tradeoff.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Pattern prompt"] --> B["Pattern orchestration"]
       B --> C["debate pattern result"]
       C --> D["Trace metadata"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `patterns` use-cases and run `debate_pattern`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/patterns/debate_pattern.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/patterns/debate_pattern.py
   :language: python
   :linenos:
