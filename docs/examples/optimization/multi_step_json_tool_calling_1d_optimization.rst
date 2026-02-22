Multi Step Json Tool Calling 1D Optimization
============================================

Source: ``examples/optimization/multi_step_json_tool_calling_1d_optimization.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/optimization/multi_step_json_tool_calling_1d_optimization.py

Motivation
----------

Run traced LLM-driven optimization with callable increase/decrease tools.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Optimization objective"] --> B["Agent selects tool"]
       B --> C["multi step json tool calling 1d optimization trajectory"]
       C --> D["Best-seen summary and trace"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `optimization` use-cases and run `multi_step_json_tool_calling_1d_optimization`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/optimization/multi_step_json_tool_calling_1d_optimization.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/optimization/multi_step_json_tool_calling_1d_optimization.py
   :language: python
   :linenos:
