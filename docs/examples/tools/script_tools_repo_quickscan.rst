Script Tools / Repo Quickscan
=============================

Source: ``examples/tools/script_tools/repo_quickscan.sh``

Introduction
------------

This script focuses on the practical so-what for tool integration: a deterministic, inspectable JSON
contract for repository quickscan operations.

Technical Implementation
------------------------

1. Read JSON input from ``stdin`` and execute the quickscan workflow.
2. Write report + trace artifacts under ``artifacts/``.
3. Emit one JSON envelope on ``stdout`` with ``ok/result/artifacts/warnings/error``.

.. mermaid::

   flowchart LR
   A["JSON input payload"] --> B["repo_quickscan.sh"]
   B --> C["report.txt + trace artifact"]
   C --> D["JSON envelope on stdout"]

.. literalinclude:: ../../../examples/tools/script_tools/repo_quickscan.sh
   :language: bash
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   bash examples/tools/script_tools/repo_quickscan.sh

- Reads one JSON object from ``stdin``.
- Prints one JSON envelope containing ``ok/result/artifacts``.
- Writes artifacts under ``artifacts/repo_quickscan`` and ``artifacts/examples/traces``.
