Derp MCP General Solver
=======================

Source: ``examples/tools/derp_mcp_general_solver.py``

Introduction
------------

This example shows the maintained DRAG + DERP path for selecting a packaged
design-research problem, exposing it as an MCP tool server, and reading solver
hints before making any optimization call. It is intentionally small: use the
DERP catalog API for problem discovery, use the packaged DERP MCP CLI for tool
launch, and keep provider-specific LLM setup outside the problem plumbing.

Technical Implementation
------------------------

1. Search DERP with ``search_problem_summaries(...)`` instead of putting full
   problem briefs into the agent prompt.
2. Load the selected optimization problem and read ``solver_hints()`` directly
   so variable domain and constraints do not need to be inferred from prose.
3. Attach the packaged DERP MCP CLI with ``MCPServerConfig.python_module(...)``.
4. Invoke ``solver_hints`` and ``evaluate`` through ``Toolbox`` and print one
   compact JSON payload.

.. mermaid::

   flowchart LR
       A["DERP search_problem_summaries"] --> B["Select problem id"]
       B --> C["DERP solver_hints()"]
       B --> D["python -m design_research_problems.mcp"]
       D --> E["DRAG Toolbox MCP tools"]
       C --> F["Agent-ready routing payload"]
       E --> F

.. literalinclude:: ../../../examples/tools/derp_mcp_general_solver.py
   :language: python
   :lines: 53-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/tools/derp_mcp_general_solver.py

When ``design-research-problems[mcp]`` is available, the example prints a JSON
object containing the selected problem summary, local solver hints, MCP tool
names, MCP solver hints, and one evaluation report. If DERP is not installed,
it exits successfully with an ``available: false`` payload and an install hint.

.. code-block:: text

   {
     "available": true,
     "example": "tools/derp_mcp_general_solver.py",
     "mcp_tools": ["drp_problem::evaluate", "drp_problem::solver_hints", "drp_problem::submit_final"],
     "problem_id": "pill_capsule_min_area"
   }

References
----------

- `design-research-problems documentation <https://cmudrc.github.io/design-research-problems/>`_
- `Model Context Protocol Specification <https://modelcontextprotocol.io/specification/2025-06-18>`_
- `SciPy optimize documentation <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_
