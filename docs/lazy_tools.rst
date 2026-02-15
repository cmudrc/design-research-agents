Lazy Tools
==========

Lazy tools are local ``.py`` or ``.sh`` scripts discovered from configured
search paths. Each script declares tool metadata in a header docblock/comment
block within the first 120 lines.

How lazy tools execute
----------------------

- Runtime sends JSON input over ``stdin``.
- Script must print exactly one JSON object on ``stdout``.
- Runtime validates the output envelope and returns a canonical ``ToolResult``.
- Policy rules still apply (command allowlist, network policy, artifact path
  restrictions).

Required directives:

- ``@tool_name``
- ``@description``
- ``@inputs``
- ``@outputs`` (must include ``stdout_json: true``)
- ``@capabilities``

Header template
---------------

.. code-block:: text

   # @tool_name: summarize_notes
   # @description: Summarize a notes file into bullet points.
   # @inputs:
   #   path: path
   #   max_points: int = 5
   # @outputs:
   #   stdout_json: true
   # @capabilities:
   #   filesystem_read: true
   #   filesystem_write: false
   #   network: false
   #   commands: []
   # @timeout_s: 20
   # @platform: [darwin, linux]
   # @version: 1.0.0

Output envelope
---------------

Lazy scripts should emit this shape:

.. code-block:: json

   {
     "ok": true,
     "result": {"summary": ["..."]},
     "artifacts": [{"path": "artifacts/example.txt", "mime": "text/plain"}],
     "warnings": [],
     "error": null
   }

Enable lazy tools in runtime config
-----------------------------------

.. code-block:: yaml

   lazy_tools:
     enabled: true
     search_paths: [examples/lazy_tools, ~/.dra/tools]
     allow_network: false
     allow_writes_outside_artifacts: false
     allowed_commands: [git, rg, python, python3, uv, ruff, pytest]
     timeout_s_default: 30

CLI helpers:

.. code-block:: bash

   dra lazy lint examples/lazy_tools
   dra lazy list --config tool_runtime.yaml
   dra lazy run lazy::rubric_score --json '{"text":"hello"}' --config tool_runtime.yaml

Programmatic usage
------------------

.. code-block:: python

   from design_research_agents import UnifiedToolRuntime

   runtime = UnifiedToolRuntime.lazy(
       search_paths=("examples/lazy_tools",),
       workspace_root=".",
       enable_core_tools=False,
   )

   result = runtime.invoke(
       "lazy::rubric_score",
       {"text": "one two three four five"},
       request_id="lazy-example",
       dependencies={},
   )

Troubleshooting
---------------

- ``dra lazy lint`` fails:
  - Check required header directives and indentation in ``@inputs``.
- ``Unknown lazy tool`` at runtime:
  - Verify ``search_paths`` and tool name prefix ``lazy::``.
- ``Lazy tool stdout must be JSON``:
  - Print logs to ``stderr`` and only emit one JSON object to ``stdout``.

See examples in:

- ``examples/lazy_tools/python/rubric_score.py``
- ``examples/lazy_tools/bash/repo_quickscan.sh``
- ``examples/lazy_tools/python/single_step_json_lazy_rubric_score_agent.py``
- ``examples/lazy_tools/bash/single_step_json_lazy_repo_quickscan_agent.py``
