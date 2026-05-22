Multi Step JSON Tool Calling 1d Optimization
============================================

Source: ``examples/optimization/multi_step_json_tool_calling_1d_optimization.py``

Introduction
------------

Practical Bayesian optimization motivates iterative search over expensive objective evaluations, while
Toolformer and Plan-and-Solve motivate explicit action/reason loops for model-guided exploration. This
example operationalizes that idea as a JSON tool-calling optimization workflow with traceable proposals and
evaluations.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["MultiStepAgent.run(...)"]
       C --> D["optimization loop combines callable tools with explicit final answers"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/optimization/multi_step_json_tool_calling_1d_optimization.py
   :language: python
   :lines: 58-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/optimization/multi_step_json_tool_calling_1d_optimization.py

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

- `Practical Bayesian Optimization of Machine Learning Algorithms <https://arxiv.org/abs/1012.2599>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
