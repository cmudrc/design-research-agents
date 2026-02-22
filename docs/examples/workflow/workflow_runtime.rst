Workflow Runtime
================

Source: ``examples/workflow/workflow_runtime.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_runtime.py

Motivation
----------

Run traced public ``Workflow`` facade for a design-check summary.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Workflow input"] --> B["Workflow steps"]
       B --> C["workflow runtime final output"]
       C --> D["Trace metadata"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `workflow` use-cases and run `workflow_runtime`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/workflow/workflow_runtime.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/workflow/workflow_runtime.py
   :language: python
   :linenos:
