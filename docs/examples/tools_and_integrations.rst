Tools and Integrations Examples
===============================

These examples exercise runtime tooling layers in design-focused tasks.

Scripts and observations
------------------------

- ``examples/tools/mcp_minimal.py``
  Observe: namespaced MCP tool inventory and direct invocation result.
  Public API: ``Toolbox``, ``McpServer``.
- ``examples/tools/source_fusion_story.py``
  Observe: combined metrics from core/script/MCP sources and report artifact path.
  Public API: ``Toolbox``, ``ScriptTool``, ``McpServer``.
- ``examples/tools/script_tools/python/rubric_score.py``
  Observe: script envelope with JSON report artifact and trace artifact path.
- ``examples/tools/script_tools/bash/repo_quickscan.sh``
  Observe: shell tool envelope with line-count result and trace artifact path.

Run examples from repository root.

Observed local run snippets (2026-02-21)
----------------------------------------

``examples/tools/source_fusion_story.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/tools/source_fusion_story.py

Observed stdout (paths redacted):

.. code-block:: json

   {
     "example": "tools/source_fusion_story.py",
     "input_path": "<repo_root>/artifacts/examples/source_fusion_story_input.txt",
     "report_path": "<repo_root>/artifacts/examples/source_fusion_story_report.json",
     "source_tool_counts": {"core": 18, "mcp": 18, "script": 1},
     "word_count_match": true
   }

``examples/tools/script_tools/bash/repo_quickscan.sh``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   bash examples/tools/script_tools/bash/repo_quickscan.sh

Observed stdout:

.. code-block:: json

   {
     "ok": true,
     "result": {
       "line_count": 25,
       "trace_path": "artifacts/examples/traces/run_<timestamp>_example-script-repo-quickscan-001.jsonl"
     },
     "error": null
   }

``examples/tools/script_tools/python/rubric_score.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=src python3 examples/tools/script_tools/python/rubric_score.py

Observed stdout (paths redacted):

.. code-block:: json

   {
     "ok": true,
     "result": {
       "score": 0,
       "max_score": 10,
       "cwd": "<repo_root>",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_example-script-rubric-score-001.jsonl"
     },
     "error": null
   }
