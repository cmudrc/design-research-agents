Dependencies and Extras
=======================

Core Install
------------

.. code-block:: bash

   pip install design-research-agents

Editable contributor setup:

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e ".[dev]"

Or use:

.. code-block:: bash

   make dev

Reproducible install
--------------------

.. code-block:: bash

   make repro REPRO_EXTRAS="dev full"

``REPRO_EXTRAS`` defaults to ``dev full``. The frozen install fails when
``uv.lock`` is out of date.

Maintainer release baseline
---------------------------

Use this when preparing a tagged release:

1. Use Python ``3.12`` (from ``.python-version``).
2. Regenerate lock data: ``make lock``.
3. Verify the frozen install and full checks: ``make repro REPRO_EXTRAS="dev full"`` and ``make ci``.
4. Build release artifacts and validate metadata: ``make release-check``.
5. Commit ``uv.lock`` (and any dependency spec changes), then tag and publish.

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

Hosted clients are the fastest path for onboarding and benchmark iteration, but
they require network access and data egress. Local in-process clients are often
preferable for privacy-sensitive studies and single-machine experimentation, but
they are more hardware-sensitive. Server-backed local clients improve deployment
flexibility and throughput isolation, but they add service-management overhead.

Recommended install profiles:

- hosted OpenAI-family studies: ``pip install -e ".[dev,openai]"``
- hosted provider comparisons: ``pip install -e ".[dev,providers]"``
- Chroma-backed memory experiments: ``pip install -e ".[dev,memory_chroma]"``
- graph-memory experiments: ``pip install -e ".[dev,memory_graph]"``
- local-only studies: ``pip install -e ".[dev,local]"``
- broad backend validation: ``pip install -e ".[dev,full]"``

Reproducible and release flows are exposed via ``make repro``, ``make lock``,
and ``make release-check``.
