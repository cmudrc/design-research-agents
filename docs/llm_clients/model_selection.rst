Model Selection
===============

Use model selection to keep client/model choices consistent under explicit
constraints.

Core API
--------

- ``ModelSelector``: flat selector interface for task/constraint inputs.
- ``ModelFlightRegistry``: named, reproducible model sets for experiments.
- ``ModelFlight``: one model-family or hosted-provider candidate set.
- ``ModelCatalog``: flattened candidate list consumed by selection policy.
- ``ModelSelector.select(..., output="decision")`` returns internal
  ``ModelSelectionDecision`` metadata.
- ``ModelSelector.select(..., output="client_config")`` returns a plain mapping
  with ``provider``, ``model_id``, ``client_class``, and constructor ``kwargs``.
- ``ModelSelector.select(..., output="client")`` (default) returns an
  instantiated LLM client.

Underlying policy/types
-----------------------

``ModelSelector`` delegates to internal policy components:

- ``ModelSelectionPolicy`` for candidate scoring and decisioning
- ``ModelSelectionIntent`` for task/priority intent
- ``ModelSelectionConstraints`` for local/provider/cost/latency bounds
- ``HardwareProfile`` for hardware-aware local fit checks

Model flights
-------------

Model flights are explicit candidate lists. They are useful when an experiment
needs to enumerate a model matrix directly instead of asking ``ModelSelector`` to
choose one model.

The default flight registry includes:

- ``qwen3-gguf``: Qwen3 instruction variants across size and GGUF quantization
- ``gemma3-gguf``: Gemma 3 instruction variants across size and GGUF quantization
- ``llama-gguf``: Llama instruction variants from small local models through large comparisons
- ``mistral-gguf``: Mistral and Mixtral instruction variants
- ``phi-gguf``: compact Phi instruction variants
- ``open-reasoning``: open-weight reasoning models such as gpt-oss, DeepSeek R1, and Phi reasoning
- ``frontier-moe-open-weights``: long-context and frontier-scale open MoE references
- ``agentic-coding-open-weights``: repository-scale coding and tool-use models
- ``vision-language-open-weights``: open VLMs for images, documents, and multimodal artifacts
- ``openai-api``: hosted OpenAI API candidates

.. code-block:: python

   from design_research_agents import ModelCatalog, ModelFlightRegistry, ModelSelector

   flights = ModelFlightRegistry.default()
   gemma = flights.require("gemma3-gguf")

   for model in gemma.models:
       print(model.model_id, model.size_b, model.quantization)

   selector = ModelSelector(catalog=ModelCatalog.from_flights([gemma]))
   decision = selector.select(task="summarize design interviews", output="decision")

SOTA coverage
-------------

The default catalog intentionally separates local GGUF size/quantization sweeps
from frontier reference flights. This keeps everyday local selection stable while
still making state-of-the-art model classes discoverable for experiment design:

- Open reasoning: OpenAI's gpt-oss family, DeepSeek R1, and Phi reasoning cover
  local/open-weight reasoning and tool-use baselines.
- Frontier MoE: Qwen3 MoE/Next, DeepSeek V3.2, Llama 4, and GLM cover
  long-context, sparse-attention, and high-capacity comparisons.
- Agentic coding: Qwen3-Coder, Kimi K2 Thinking, and MiniMax-M2 cover
  repository-scale coding, tool orchestration, and long-horizon agent workflows.
- Vision-language: Gemma 3, Gemma 3n, Qwen3-VL, and Phi-4 reasoning vision cover
  image/document understanding and multimodal design artifacts.

This coverage is based on primary model-card and release materials from
`OpenAI gpt-oss <https://openai.com/index/introducing-gpt-oss/>`_,
`DeepSeek R1-0528 <https://huggingface.co/deepseek-ai/DeepSeek-R1-0528>`_,
`DeepSeek V3.2 <https://huggingface.co/deepseek-ai/DeepSeek-V3.2>`_,
`Qwen3-235B <https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507>`_,
`Qwen3-Next <https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct>`_,
`Qwen3-Coder <https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct>`_,
`Qwen3-VL <https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct>`_,
`Llama 4 <https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct>`_,
`Gemma 3 <https://ai.google.dev/gemma/docs/core/model_card_3>`_,
`Phi-4 reasoning vision <https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B>`_,
`Kimi K2 Thinking <https://huggingface.co/moonshotai/Kimi-K2-Thinking>`_,
`GLM-4.5 <https://arxiv.org/abs/2508.06471>`_, and
`MiniMax-M2 <https://huggingface.co/MiniMaxAI/MiniMax-M2>`_.

Model catalogs
--------------

``ModelCatalog`` is the flattened inventory that selection consumes. Flights are
reproducible experiment sets; catalogs are the queryable model inventory. Use
catalog helpers to inspect, narrow, or combine candidate pools before routing.

.. code-block:: python

   from design_research_agents import ModelCatalog, ModelFlightRegistry

   registry = ModelFlightRegistry.default()
   catalog = ModelCatalog.default()
   local_gemma = catalog.by_family("gemma3").filter(quantization="q4_k_m")
   hosted = catalog.remote()
   experiment_catalog = ModelCatalog.from_flights(
       [
           registry.require("qwen3-gguf"),
           registry.require("gemma3-gguf"),
       ]
   )

   print(local_gemma.model_ids())
   print(hosted.model_ids())
   print(experiment_catalog.signature())

Catalog entries carry provenance and discovery metadata such as ``source``,
``repo_id``, ``artifact``, ``license``, ``context_window``, ``capabilities``, and
``tags``. The default catalog is curated and deterministic. Additional catalogs
can be merged explicitly, with duplicate model ids rejected unless
``replace=True`` is passed.

Hugging Face catalog discovery
------------------------------

Install the optional Hugging Face metadata dependency only when catalog discovery
needs to call the Hub:

.. code-block:: bash

   pip install -e ".[huggingface]"

Then build a catalog from Hub metadata:

.. code-block:: python

   from design_research_agents import ModelCatalog

   hf_catalog = ModelCatalog.from_huggingface(
       ["google/gemma-3-4b-it"],
       provider="llama_cpp",
       revision="main",
       capabilities=("chat", "local"),
       tags=("candidate",),
   )

``ModelCatalog.from_huggingface`` is lazy: it imports ``huggingface_hub`` only
when called without an injected ``api`` object. Tests and deterministic pipelines
can pass a small object exposing ``model_info(...)`` to avoid network I/O.

Attribution note
----------------

The catalog direction is informed by `llmfit <https://www.llmfit.org/>`_, which
publicly emphasizes hardware-aware model selection, a curated Hugging Face model
database, dynamic quantization selection, runtime-provider awareness, and
fit/speed/context scoring. The implementation here does not vendor llmfit code
or data and does not depend on llmfit at runtime. llmfit's public repository is
`AlexsJones/llmfit <https://github.com/AlexsJones/llmfit>`_.

Constraint examples
-------------------

Local-only selection:

.. code-block:: python

   from design_research_agents import ModelSelector

   selector = ModelSelector()
   decision = selector.select(
       task="summarize interview notes",
       priority="balanced",
       require_local=True,
       output="decision",
   )

Cost-capped selection:

.. code-block:: python

   selector = ModelSelector()
   decision = selector.select(task="summarize", max_cost_usd=0.01, output="decision")

Latency-capped selection:

.. code-block:: python

   selector = ModelSelector()
   decision = selector.select(task="chat", max_latency_ms=1200, output="decision")

Local model fit behavior
------------------------

``ModelSelector`` evaluates local candidates against available RAM/VRAM hints
and can prefer remote candidates under high system load. This preserves a
local-first posture while avoiding local models likely to overload the current
machine.

See examples
------------

- ``examples/model_selection/local.py``
- ``examples/model_selection/remote.py``
- ``examples/model_selection/README.md``

Related
-------

- :doc:`index`
