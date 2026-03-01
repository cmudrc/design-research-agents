TransformersLocalLLMClient
==========================

``TransformersLocalLLMClient`` runs inference locally via the transformers
stack.

Default behavior
----------------

- Default model id and model name: ``distilgpt2``
- Default device policy: ``auto``
- Local in-process execution

Constructor-first usage
-----------------------

.. code-block:: python

   from design_research_agents import TransformersLocalLLMClient
   from design_research_agents.llm import LLMMessage, LLMRequest

   client = TransformersLocalLLMClient(model_id="distilgpt2")
   response = client.generate(
       LLMRequest(
           messages=(LLMMessage(role="user", content="Summarize this transcript section."),),
           model=client.default_model(),
       )
   )

Dependencies and environment
----------------------------

- Install transformers backend extras: ``pip install -e \".[transformers]\"``
- Sufficient local CPU/GPU memory for selected model

Model notes for local runs
--------------------------

- Start with smaller instruct checkpoints for fast iteration and lower memory
  pressure.
- Move to larger checkpoints only when quality gains justify latency/footprint.
- Keep ``default_model`` aligned with the checkpoint your runtime can sustain
  for repeated workflow runs.

Examples
--------

- ``examples/clients/transformers_local_client.py``

Attribution
-----------

- Docs: `Hugging Face Transformers docs <https://huggingface.co/docs/transformers/index>`_
- Homepage: `Hugging Face <https://huggingface.co/>`_
