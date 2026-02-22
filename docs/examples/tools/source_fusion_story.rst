Source Fusion Story
===================

Source: ``examples/tools/source_fusion_story.py``

Introduction
------------

MCP standardizes tool connectivity, data-fusion concepts motivate combining heterogeneous signals, and RAG
provides a grounding mechanism for synthesis over retrieved evidence. This example fuses MCP tools and
script tools into one workflow that emits a traceable narrative artifact.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Toolbox.invoke_dict(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["Toolbox.invoke_dict(...)"]
       C --> D["core, script, and MCP tools execute in one composed runtime"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/tools/source_fusion_story.py
   :language: python
   :lines: 63-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/tools/source_fusion_story.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "core_word_count": 14,
     "example": "tools/source_fusion_story.py",
     "input_path": "artifacts/examples/<truncated-input-path>",
     "mcp_word_count": 14,
     "report_path": "artifacts/examples/<truncated-report-path>",
     "score_percent": 10.0,
     "script_max_score": 20,
     "script_score": 2,
     "script_trace_path": "artifacts/examples/traces/run_20260222T162210Z_example-script-rubric-score-001.jsonl",
     "source_tool_counts": {
       "core": 23,
       "mcp": 23,
       "script": 1
     },
     "trace": {
       "request_id": "example-tools-source-fusion-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-tools-source-fusion-design-001.jsonl"
     },
     "word_count_match": true
   }

References
----------

- `Model Context Protocol Specification <https://modelcontextprotocol.io/specification/2025-06-18>`_
- `Data Fusion (Wikipedia) <https://en.wikipedia.org/wiki/Data_fusion>`_
- `Retrieval-Augmented Generation <https://arxiv.org/abs/2005.11401>`_
