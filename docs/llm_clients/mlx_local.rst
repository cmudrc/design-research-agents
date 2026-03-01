MLXLocalLLMClient
=================

``MLXLocalLLMClient`` runs local inference on Apple MLX.

Default behavior
----------------

- Default model id and model name:
  ``mlx-community/Qwen2.5-1.5B-Instruct-4bit``
- Local execution optimized for Apple silicon

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import MLXLocalLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = MLXLocalLLMClient()
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Produce three concise insights."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Install MLX backend extras: ``pip install -e \".[mlx]\"``
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

Attribution
-----------

- Docs: `MLX docs <https://ml-explore.github.io/mlx/build/html/index.html>`_
- Homepage: `MLX GitHub <https://github.com/ml-explore/mlx>`_
