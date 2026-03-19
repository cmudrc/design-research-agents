Dependencies and Extras
=======================

The project supports a minimal core install plus optional extras for backend coverage.

Core install
------------

.. code-block:: bash

   pip install -e .

Development install
-------------------

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
     - Key packages
     - Platform constraints
   * - ``dev``
     - Linting, typing, tests, docs, and release checks
     - ``pytest``, ``ruff``, ``mypy``, ``sphinx``, ``build``, ``twine``
     - None
   * - ``hf``
     - Shared Hugging Face utilities
     - ``huggingface-hub``
     - None
   * - ``anthropic``
     - Anthropic hosted backend
     - ``anthropic``
     - None
   * - ``gemini``
     - Gemini hosted backend
     - ``google-genai``
     - None
   * - ``groq``
     - Groq hosted backend
     - ``groq``
     - None
   * - ``llama_cpp``
     - Managed llama.cpp server backend
     - ``llama-cpp-python[server]``, ``huggingface-hub``
     - None
   * - ``mlx``
     - Apple MLX backend
     - ``mlx-lm``
     - macOS + Apple Silicon only
   * - ``transformers``
     - In-process transformers backend
     - ``torch``, ``transformers``, ``huggingface-hub``
     - None
   * - ``vllm``
     - vLLM server backend
     - ``vllm``
     - Linux only, Python < 3.14
   * - ``sglang``
     - SGLang server backend
     - ``sglang``
     - Linux only, Python < 3.14
   * - ``local``
     - Convenience bundle for local backends
     - ``llama_cpp`` + ``mlx`` + ``transformers`` stack
     - ``mlx`` portion only on macOS + Apple Silicon
   * - ``full``
     - Convenience bundle for local + server backends
     - ``local`` + ``vllm`` + ``sglang`` stack
     - ``vllm``/``sglang`` portions only on Linux, Python < 3.14

Recommended profiles
--------------------

- Fast contributor loop: ``make dev``
- Anthropic hosted experimentation: ``pip install -e ".[dev,anthropic]"``
- Gemini hosted experimentation: ``pip install -e ".[dev,gemini]"``
- Groq hosted experimentation: ``pip install -e ".[dev,groq]"``
- Local backend experimentation: ``pip install -e ".[dev,local]"``
- Full backend coverage (where supported): ``pip install -e ".[dev,full]"``
- Minimal hosted-only setup: ``pip install -e .``

Notes
-----

- Extras are additive and can be combined.
- Linux-only extras are safely skipped by pip on unsupported platforms.
- For deterministic example tests, use the monkeypatch harness under
  ``tests/example_monkeypatch`` rather than adding deterministic logic to examples.
