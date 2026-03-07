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

Extras Matrix
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
- local-only studies: ``pip install -e ".[dev,local]"``
- broad backend validation: ``pip install -e ".[dev,full]"``

Reproducible and release flows are exposed via ``make repro``, ``make lock``,
and ``make release-check``.
