Dependencies and Extras
=======================

Core Install
------------

.. code-block:: bash

   python -m pip install design-research-agents

Editable contributor setup:

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

Or use:

.. code-block:: bash

   make dev

Maintainer release baseline
---------------------------

Use this when preparing a tagged release:

1. Use Python ``3.12`` (from ``.python-version``).
2. Install maintainer dependencies: ``make dev``.
3. Run the automated CI baseline: ``make ci``.
4. Build the public documentation: ``make docs-build``.
5. When links or navigation changed, run ``make docs-linkcheck``.
6. Build release artifacts and validate metadata: ``make release-check``.
7. Commit the reviewed release metadata and documentation, then tag and publish.

``make release-check`` builds both the source distribution and wheel into ``dist/``
and runs ``twine check`` against the generated artifacts.

Extras matrix
-------------

.. list-table::
   :header-rows: 1

   * - Extra
     - Purpose
   * - ``dev``
     - Contributor tooling
   * - ``openai`` / ``azure``
     - OpenAI-family hosted SDK backends
   * - ``anthropic``
     - Anthropic hosted backend
   * - ``gemini``
     - Gemini hosted backend
   * - ``groq``
     - Groq hosted backend
   * - ``mcp``
     - MCP tool-runtime integration
   * - ``huggingface``
     - Hugging Face Hub metadata discovery for ``ModelCatalog.from_huggingface``
   * - ``memory_chroma``
     - Optional ChromaDB-backed vector memory store
   * - ``memory_graph``
     - Optional NetworkX-backed graph memory store
   * - ``llama_cpp``
     - Managed llama.cpp backend
   * - ``transformers``
     - In-process transformers backend
   * - ``mlx``
     - Apple MLX backend
   * - ``vllm``
     - vLLM server backend (Linux)
   * - ``sglang``
     - SGLang server backend (Linux)
   * - ``local``
     - Local-backend convenience bundle
   * - ``providers``
     - Hosted-provider convenience bundle
   * - ``full``
     - Providers + local backends
   * - ``all``
     - ``full`` plus optional ChromaDB and graph-memory stores

``full`` remains the backend-focused bundle. Use ``all`` when you want that same
runtime surface plus the optional memory backends exposed by this package.

Hosted clients are the fastest path for onboarding and benchmark iteration, but
they require network access and data egress. Local in-process clients are often
preferable for privacy-sensitive studies and single-machine experimentation, but
they are more hardware-sensitive. Server-backed local clients improve deployment
flexibility and throughput isolation, but they add service-management overhead.

Recommended install profiles:

- hosted OpenAI-family studies:
  ``python -m pip install "design-research-agents[openai]"``
- hosted provider comparisons:
  ``python -m pip install "design-research-agents[providers]"``
- DRAG/DERP MCP workflows:
  ``python -m pip install "design-research-agents[mcp]" "design-research-problems[mcp]"``
- Hugging Face catalog discovery:
  ``python -m pip install "design-research-agents[huggingface]"``
- Chroma-backed memory experiments:
  ``python -m pip install "design-research-agents[memory_chroma]"``
- graph-memory experiments:
  ``python -m pip install "design-research-agents[memory_graph]"``
- local-only studies: ``python -m pip install "design-research-agents[local]"``
- broad backend validation: ``python -m pip install "design-research-agents[full]"``
- broad runtime + memory validation:
  ``python -m pip install "design-research-agents[all]"``

From a source checkout, replace ``design-research-agents`` with ``.`` and add
``-e``. Add the ``dev`` extra separately when running contributor checks; it is
not required by these runtime profiles.

Release validation is exposed via ``make release-check``.
