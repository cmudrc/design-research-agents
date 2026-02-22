Source Fusion Story
===================

Source: ``examples/tools/source_fusion_story.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/tools/source_fusion_story.py

Motivation
----------

Run traced source-fusion runtime example across core/script/MCP tool sources.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Tool input"] --> B["Tool runtime"]
       B --> C["source fusion story result"]
       C --> D["Artifacts and trace"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `tools` use-cases and run `source_fusion_story`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/tools/source_fusion_story.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/tools/source_fusion_story.py
   :language: python
   :linenos:
