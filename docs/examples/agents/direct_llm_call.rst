Direct LLM Call
===============

Source: ``examples/agents/direct_llm_call.py``

Introduction
------------

Engineering-design studies show that transparent prompt-to-response traces are essential for credible
evaluation and human oversight; the benchmark framing in Toward Engineering AGI and the collaboration
framing in Human-AI collaboration by design both depend on this visibility, while llama.cpp server docs
ground practical local deployment. This example is the smallest reproducible path for observing one direct
call end to end with runtime traces.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``DirectLLMCall.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["DirectLLMCall.run(...)"]
       C --> D["WorkflowRuntime executes one direct call"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/agents/direct_llm_call.py
   :language: python
   :lines: 57-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/direct_llm_call.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "error": null,
     "example": "agents/direct_llm_call.py",
     "final_output": "4",
     "model": "example-model",
     "package_version": "0.2.0",
     "success": true,
     "terminated_reason": null,
     "trace": {
       "request_id": "example-direct-llm-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162205Z_example-direct-llm-design-001.jsonl"
     }
   }

References
----------

- `Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs <https://arxiv.org/abs/2509.16204>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
- `llama.cpp llama-server docs <https://github.com/ggml-org/llama.cpp#llama-server>`_
