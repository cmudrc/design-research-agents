Examples Guide
==============

The examples in this repository are runnable research-oriented scripts. They are
designed to show not only API usage, but how the library fits into realistic
experimental workflows. Each example lists dependencies, expected scope, and
the primary concept it demonstrates.

Featured Examples
-----------------

Direct LLM Call
~~~~~~~~~~~~~~~~

One-step participant execution with a configured backend client.

**Requires:** base install + reachable backend endpoint
**Runtime:** short
**Teaches:** baseline participant setup, request execution, structured output handling

Multi-Step JSON Tool Calling Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterative tool-using execution with explicit action/observation loops.

**Requires:** base install
**Runtime:** short to medium
**Teaches:** tool-routing behavior, multi-step control, inspectable intermediate state

Debate Pattern
~~~~~~~~~~~~~~

Role-based multi-agent coordination with adjudication workflow structure.

**Requires:** base install
**Runtime:** medium
**Teaches:** orchestration patterns, delegate coordination, traceable multi-role reasoning

MCP Minimal
~~~~~~~~~~~

Small end-to-end MCP-backed tool integration example.

**Requires:** ``mcp``-compatible server/runtime setup
**Runtime:** medium
**Teaches:** external tool connectivity, MCP source wiring, runtime safety boundaries

Deterministic runs for tests are provided by
``tests/example_monkeypatch/sitecustomize.py`` when
``DRA_EXAMPLE_LLM_MODE=deterministic`` is set.

Full Catalog
------------

.. toctree::
   :maxdepth: 2

   agents/index
   workflow/index
   patterns/index
   clients/index
   model_selection/index
   tools/index
   optimization/index
