Examples Guide
==============

This section describes runnable examples, their intent, and what results to
observe after execution.

All runnable examples are design-focused and trace-enabled.
By default, examples emit JSONL traces under ``artifacts/examples/traces``.
Output snippets in this guide were captured from local runs on
``2026-02-21``.

Local run mode for reproducible docs output
-------------------------------------------

Use deterministic mode for stable example output and no external model calls:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/agents/basic/direct_llm_call.py

Sensitive-data guardrails used in this guide:

- No API keys, tokens, or hostnames are shown.
- Absolute workspace paths are redacted as ``<repo_root>/...``.
- Trace filenames are shown with ``<timestamp>`` placeholders where useful.

Pages
-----

- :doc:`agents`
- :doc:`workflows`
- :doc:`tools_and_integrations`
- :doc:`clients_and_selection`
- :doc:`trace_observability`

.. toctree::
   :maxdepth: 2

   agents
   workflows
   tools_and_integrations
   clients_and_selection
   trace_observability
