Nominal Team
============

Source: ``examples/patterns/nominal_team.py``

Introduction
------------

Nominal teams explore one task independently, then hand all candidate outputs to a dedicated
evaluator for best-of-N selection. This example fans out a design prompt to three focused
contributors and selects the strongest result with a structured evaluator response.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build three focused ``DirectLLMCall`` delegates and one evaluator over a shared ``LlamaCppServerLLMClient``.
3. Execute ``NominalTeamPattern.run(...)`` with member-specific prompt templates for diverse independent drafts.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["NominalTeamPattern.run(...)"]
       B --> C["repairability / reliability / manufacturability members generate independently"]
       C --> D["evaluator compares candidates and selects best member"]
       D --> E["ExecutionResult/payload"]
       E --> F["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/nominal_team.py
   :language: python
   :lines: 49-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/nominal_team.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<selected-candidate-payload>",
     "terminated_reason": "<string-or-null>",
     "error": null,
     "trace": {
       "request_id": "<request-id>",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }

References
----------

- `Self-Consistency Improves Chain of Thought Reasoning in Language Models <https://arxiv.org/abs/2203.11171>`_
- `Tree of Thoughts: Deliberate Problem Solving with Large Language Models <https://arxiv.org/abs/2305.10601>`_
- `Nominal group technique <https://en.wikipedia.org/wiki/Nominal_group_technique>`_
