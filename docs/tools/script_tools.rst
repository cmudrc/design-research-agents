Script Tools
============

Script tools are explicit ``ScriptTool`` entries passed to ``Toolbox`` or loaded
from ``script_tools`` runtime config. Each tool points to a local ``.py`` or ``.sh``
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

Programmatic helpers
--------------------

.. code-block:: python

   from design_research_agents import ScriptTool, Toolbox

   runtime = Toolbox(
       script_tools=(
           ScriptTool(
               name="rubric_score",
               path="examples/tools/script_tools/rubric_score.py",
               description="Score text with a simple rubric.",
           ),
       )
   )
   names = [spec.name for spec in runtime.list_tools() if spec.metadata.source == "script"]
   result = runtime.invoke(
       "script::rubric_score",
       {"text": "hello"},
       request_id="docs-script",
       dependencies={},
   )
   runtime.close()

Troubleshooting
---------------

- Script tool missing from ``Toolbox.list_tools()``: verify ``script_tools`` config and script path.
- ``Unknown script tool``: verify ``script::`` prefix and canonical tool name.
- ``stdout must be JSON``: log to stderr, emit one JSON object on stdout.

Examples
--------

- ``examples/tools/script_tools/README.md``
- ``examples/tools/script_tools/rubric_score.py``
- ``examples/tools/script_tools/repo_quickscan.sh``
