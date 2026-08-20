Examples
========

The examples in this repository are runnable research-oriented scripts. They are
designed to show not only API usage, but how the library fits into realistic
experimental workflows. The featured examples below list dependencies,
expected scope, and the primary concept they demonstrate.

.. note::

   Some local ``LlamaCppServerLLMClient`` examples intentionally use
   ``Qwen3-4B`` GGUF configs, which can exceed available RAM on smaller
   machines. If you want a lighter local starting point, swap in a smaller
   model or begin with :doc:`clients/ollama_local_client`, which documents the
   lighter Ollama defaults.

Featured Examples
-----------------

VS Code Hello World
~~~~~~~~~~~~~~~~~~~

``examples/agents/vscode_hello_world.py`` is a deterministic first agent run
with a local stub client.

**Requires:** base install only; no model server, model download, or API key
**Runtime:** short
**Teaches:** minimal runtime client duck type, ``DirectLLMCall``, structured output

Direct LLM Call
~~~~~~~~~~~~~~~~

``examples/agents/direct_llm_call.py`` performs one-step participant execution
against a configured OpenAI-compatible HTTP backend.

**Requires:** base install + reachable backend endpoint
**Runtime:** short
**Teaches:** baseline participant setup, request execution, structured output handling

Multi-Step JSON Tool Calling Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/agents/multi_step_json_tool_calling_agent.py`` demonstrates iterative
tool-using execution with explicit action/observation loops.

**Requires:** ``llama_cpp`` extra, model download, and sufficient local memory
**Runtime:** short to medium
**Teaches:** tool-routing behavior, multi-step control, inspectable intermediate state

Debate Pattern
~~~~~~~~~~~~~~

``examples/patterns/debate_pattern.py`` demonstrates role-based multi-agent
coordination with adjudication workflow structure.

**Requires:** ``llama_cpp`` extra, model download, and sufficient local memory
**Runtime:** medium
**Teaches:** orchestration patterns, delegate coordination, traceable multi-role reasoning

MCP Minimal
~~~~~~~~~~~

``examples/tools/mcp_minimal.py`` is a small end-to-end MCP-backed tool
integration example.

**Requires:** ``mcp`` extra; the example launches the packaged local stdio server
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
