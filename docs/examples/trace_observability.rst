Trace Observability in Examples
===============================

All runnable examples emit trace metadata and write JSONL traces.

Trace location
--------------

- Default trace directory: ``artifacts/examples/traces``.
- Example payloads include ``trace.request_id`` and ``trace.trace_path``.

What to verify after running an example
---------------------------------------

1. ``trace.trace_path`` is present and points to an existing file.
2. The trace file contains newline-delimited JSON events.
3. Run-level fields include request id and success/failure metadata.

Concrete local check (2026-02-21)
---------------------------------

Example run:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/agents/basic/direct_llm_call.py

Observed stdout excerpt:

.. code-block:: json

   {
     "success": true,
     "trace": {
       "request_id": "example-direct-llm-design-001",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_example-direct-llm-design-001.jsonl"
     }
   }

Then inspect the trace file:

.. code-block:: bash

   head -n 2 artifacts/examples/traces/run_<timestamp>_example-direct-llm-design-001.jsonl

Observed trace-log excerpt:

.. code-block:: json

   {"event_type":"RunStarted","run_id":"example-direct-llm-design-001","attributes":{"agent":"DirectLLMCall"}}
   {"event_type":"AgentRunStarted","run_id":"example-direct-llm-design-001","attributes":{"agent":"WorkflowRuntime"}}

This observability contract applies to:

- Agent examples
- Workflow examples
- Tool-runtime examples
- Client/model-selection config examples
- Script-tool examples (Python and shell)

Quick Trace Analysis Harness
----------------------------

You can summarize trace metrics with the built-in analysis harness:

.. code-block:: bash

   PYTHONPATH=src python3 scripts/analyze_traces.py \
     --trace-dir artifacts/examples/traces

JSON output is also available:

.. code-block:: bash

   PYTHONPATH=src python3 scripts/analyze_traces.py \
     --trace-dir artifacts/examples/traces \
     --json

Default summary covers:

- run success/failure counts
- event counts by type
- model calls and token totals
- tool invocation success/failure counts
- workflow step status counts
- latency summaries and top errors
