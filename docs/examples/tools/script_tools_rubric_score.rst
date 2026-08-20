Script Tools / Rubric Score
===========================

Source: ``examples/tools/script_tools/rubric_score.py``

Introduction
------------

This script focuses on the practical so-what for tool integration: a deterministic, inspectable JSON
contract that keeps scripted scoring auditable in agent workflows.

Technical Implementation
------------------------

1. Read one JSON payload from ``stdin`` and compute rubric score fields.
2. Write report + trace artifacts under ``artifacts/``.
3. Emit one JSON envelope on ``stdout`` with ``ok/result/artifacts/warnings/error``.

.. mermaid::

   flowchart LR
       A["JSON input payload"] --> B["rubric_score.py"]
       B --> C["Score + artifacts"]
       C --> D["JSON envelope on stdout"]

.. literalinclude:: ../../../examples/tools/script_tools/rubric_score.py
   :language: python
   :lines: 25-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/tools/script_tools/rubric_score.py

- Reads one JSON object from ``stdin``.
- Prints one JSON envelope containing ``ok/result/artifacts``.
- Writes artifacts under ``artifacts/rubric_score`` and ``artifacts/examples/traces``.
