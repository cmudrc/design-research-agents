MlxLocalLLMClient
=================

``MlxLocalLLMClient`` runs local inference on Apple MLX.

Default behavior
----------------

- Default model id and model name:
  ``mlx-community/Qwen2.5-1.5B-Instruct-4bit``
- Local execution optimized for Apple silicon

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import MlxLocalLLMClient
   from design_research_agents.contracts import LLMChatParams, LLMMessage

   client = MlxLocalLLMClient()
   response = client.chat(
       messages=[LLMMessage(role="user", content="Produce three concise insights.")],
       model=client.default_model(),
       params=LLMChatParams(),
   )

Dependencies and environment
----------------------------

- Install local extras: ``pip install -e \".[local]\"``
- Apple silicon environment with MLX stack available

Model notes for local runs
--------------------------

- Prefer quantized MLX-ready instruct checkpoints for better on-device
  throughput.
- Treat model size as a latency/quality dial; validate with representative
  prompts before scaling up.
- Pair with :doc:`model_selection` when you need hardware-aware fallback
  behavior.

Examples
--------

- ``examples/clients/mlx_local_client.py``

Official docs
-------------

- `MLX <https://ml-explore.github.io/mlx/build/html/index.html>`_
