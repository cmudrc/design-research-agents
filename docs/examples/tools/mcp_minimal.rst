MCP Minimal
===========

Source: ``examples/tools/mcp_minimal.py``

Introduction
------------

MCP and JSON-RPC define interoperable tool transport contracts, and Toolformer motivates why model behavior
improves when tool calls are explicit and machine-checked. This example provides the minimal MCP
server/toolbox path for validating protocol-level integration inside the framework runtime.

Technical Implementation
------------------------

1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Toolbox.invoke(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

.. mermaid::

   flowchart LR
       A["Input prompt or scenario"] --> B["main(): runtime wiring"]
       B --> C["Toolbox.invoke(...)"]
       C --> D["MCP server registration exposes namespaced tool contracts"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/tools/mcp_minimal.py
   :language: python
   :lines: 58-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/tools/mcp_minimal.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "direct_result": 2,
     "example": "tools/mcp_minimal.py",
     "mcp_tool_count": 23,
     "sample_tools": [
       "local_core::bash.exec",
       "local_core::data.describe",
       "local_core::data.load_csv",
       "local_core::eval.decision_matrix",
       "local_core::eval.pairwise_rank"
     ],
     "trace": {
       "request_id": "example-tools-mcp-minimal-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162209Z_example-tools-mcp-minimal-design-001.jsonl"
     }
   }

References
----------

- `Model Context Protocol Specification <https://modelcontextprotocol.io/specification/2025-06-18>`_
- `JSON-RPC 2.0 Specification <https://www.jsonrpc.org/specification>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
