Multi Step JSON Tool Calling Agent
==================================

Source: ``examples/agents/multi_step_json_tool_calling_agent.py``

Introduction
------------

Toolformer motivates tool-use planning, JSON Schema defines stable machine-readable contracts, and OpenAI
function-calling guidance captures operational patterns for structured tool dispatch. This example shows a
JSON-mode agent that repeatedly selects tools through explicit schema-constrained payloads.

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
       C --> D["WorkflowRuntime loop enforces continuation and max-step policy"]
       C --> E["Tracer JSONL + console events"]
       D --> F["ExecutionResult/payload"]
       E --> F
       F --> G["Printed JSON output"]

.. literalinclude:: ../../../examples/agents/multi_step_json_tool_calling_agent.py
   :language: python
   :lines: 79-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/multi_step_json_tool_calling_agent.py

Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "allowed_tools": [
       "fs.read_text",
       "text.word_count",
       "python.sandbox",
       "memory.search",
       "memory.write",
       "memory.stats",
       "eval.decision_matrix",
       "eval.pairwise_rank"
     ],
     "error": null,
     "example": "agents/multi_step_json_tool_calling_agent.py",
     "final_output": {
       "path": "/Users/work/PycharmProjects/design-research-agents/README.md",
       "size_bytes": 800,
       "text": "# design-research-agents
   [![CI](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
   [![Coverage](.github/badges/coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
   [![Examples
   Passing](.github/badges/examples-passing.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
   [![Public API In
   Examples](.github/badges/examples-api-coverage.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/ci.yml)
   [![Docs](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml/badge.svg)](https://github.com/cmudrc/design-research-agents/actions/workflows/docs-pages.yml)

   A modular framework for",
       "truncated": true
     },
     "steps_executed": 1,
     "success": true,
     "terminated_reason": "continuation_stopped:model",
     "tool_results_count": 1,
     "trace": {
       "request_id": "example-multi-step-json-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162205Z_example-multi-step-json-design-001.jsonl"
     }
   }

References
----------

- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
