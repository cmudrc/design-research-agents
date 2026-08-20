Direct LLM Compiled Execution
=============================

Source: ``examples/agents/direct_llm_compiled_execution.py``

Introduction
------------

Compiled delegate execution is useful when you want to inspect the bound workflow and tracing metadata
before running it. This example makes the intermediate ``CompiledExecution`` object explicit so the compile
step itself is visible and testable.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``DirectLLMCall.compile(...)`` to obtain
   ``CompiledExecution``.
3. Validate the compiled wrapper shape, then call ``CompiledExecution.run()`` with the bound request metadata.
4. Print a compact JSON payload that includes compile metadata alongside the normalized execution summary.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["DirectLLMCall.compile(...)"]
       C --> D["CompiledExecution"]
       D --> E["CompiledExecution.run()"]
       E --> F["ExecutionResult/payload"]
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/agents/direct_llm_compiled_execution.py
   :language: python
   :lines: 53-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/agents/direct_llm_compiled_execution.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "compiled_delegate_name": "DirectLLMCall",
     "compiled_request_id": "<request-id>",
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

- `Partial evaluation (Wikipedia) <https://en.wikipedia.org/wiki/Partial_evaluation>`_
- `Builder pattern (Wikipedia) <https://en.wikipedia.org/wiki/Builder_pattern>`_
- `Interpreter pattern (Wikipedia) <https://en.wikipedia.org/wiki/Interpreter_pattern>`_
