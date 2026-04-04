Ralph Loop
==========

Source: ``examples/patterns/ralph_loop.py``

Introduction
------------

Ralph loops are role-programmed, not fixed two-role propose/critic cycles: each round executes
an ordered role lineup, then a dedicated evaluator decides whether consensus quality is high enough.
This example demonstrates a four-role configuration with synthesis selection and threshold stopping.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build role-specific delegates with ``DirectLLMCall`` over one managed ``LlamaCppServerLLMClient``.
3. Execute ``RalphLoopPattern.run(...)`` with dynamic roles, evaluator role id, and typed ``LoopConfig``.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["RalphLoopPattern.run(...)"]
       C --> D["role batch executes proposer/critic/synthesizer/evaluator each round"]
       C --> E["evaluator score compared to consensus threshold"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/patterns/ralph_loop.py
   :language: python
   :lines: 57-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/patterns/ralph_loop.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
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

- `CAMEL: Communicative Agents for Mind Exploration <https://arxiv.org/abs/2303.17760>`_
- `MetaGPT <https://arxiv.org/abs/2308.00352>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
