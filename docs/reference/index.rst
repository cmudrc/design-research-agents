Module Reference
================

This section provides module-level documentation for both compatibility-guaranteed
public facades and internal implementation modules that are useful to contributors.

Public usage should prefer the curated top-level exports in :doc:`/api`.

Guaranteed Public Modules
-------------------------

.. toctree::
   :maxdepth: 2

   agent
   workflow
   patterns
   llm
   memory
   model_selection
   skills
   study
   tools

Internal Modules (Underscored, Unstable)
----------------------------------------

These modules are documented for contributor visibility but are intentionally
internal and may change without compatibility guarantees.

.. toctree::
   :maxdepth: 2

   contracts
   tracing
   prompts
   schemas
   mcp_server
   shared
