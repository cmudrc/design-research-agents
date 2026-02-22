Script Tools / Bash / Repo Quickscan
====================================

Source: ``examples/tools/script_tools/bash/repo_quickscan.sh``

Run Command
-----------

.. code-block:: bash

   bash examples/tools/script_tools/bash/repo_quickscan.sh

Motivation
----------

Produce a quick repository inventory as a runnable shell-script tool example.

Diagram
-------

.. mermaid::

   flowchart LR
   A["Tool input"] --> B["Shell quickscan"]
   B --> C["Report artifact and trace"]

Technical Walkthrough
---------------------

1. Read repository directory metadata via `ls -la`.
2. Write a plain-text artifact and emit a deterministic trace record.
3. Print one JSON tool envelope to stdout.

Expected Results
----------------

- The script exits successfully and prints one JSON object.
- The output includes `line_count` and `trace_path`.
- Artifacts are written under `artifacts/repo_quickscan` and `artifacts/examples/traces`.

Discussion
----------

This script is capability-first; deterministic behavior for full examples runs is handled in tests via monkeypatching, not by branching in this script.

@tool_name: repo_quickscan
@description: Produce a quick repository inventory snapshot.
@inputs:
include_hidden: bool = false
@outputs:
stdout_json: true
@capabilities:
filesystem_read: true
filesystem_write: true
network: false
commands: []
@timeout_s: 20
@platform: [darwin, linux]
@version: 1.1.0
@examples:
- bash repo_quickscan.sh

Source Code
-----------

.. literalinclude:: ../../../examples/tools/script_tools/bash/repo_quickscan.sh
   :language: bash
   :linenos:
