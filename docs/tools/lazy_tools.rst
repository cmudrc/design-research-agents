Script Tools
============

Script tools are explicit ``ScriptTool`` entries passed to ``Toolbox`` or loaded
from ``script_tools`` YAML config. Each tool points to a local ``.py`` or ``.sh``
script and uses a JSON stdin/stdout envelope contract.

Execution contract
------------------

- Runtime sends JSON input over ``stdin``.
- Script prints exactly one JSON object on ``stdout``.
- Runtime validates output into canonical ``ToolResult``.

Envelope shape
--------------

Scripts must print one JSON object with keys:

- ``ok``
- ``result``
- ``artifacts``
- ``warnings``
- ``error`` (optional)

CLI helpers
-----------

.. code-block:: bash

   dra script lint examples/lazy_tools
   dra script list --config tool_runtime.yaml
   dra script run script::rubric_score --json '{"text":"hello"}' --config tool_runtime.yaml

Troubleshooting
---------------

- ``dra script lint`` fails: verify script path and extension.
- ``Unknown script tool``: verify ``script_tools`` config and ``script::`` prefix.
- ``stdout must be JSON``: log to stderr, emit one JSON object on stdout.

Examples
--------

- ``examples/lazy_tools/README.md``
- ``examples/lazy_tools/python/rubric_score.py``
- ``examples/lazy_tools/bash/repo_quickscan.sh``
