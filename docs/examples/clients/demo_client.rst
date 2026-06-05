Demo Client
===========

Source: ``examples/clients/demo_client.py``

Introduction
------------

``DemoLLMClient`` is the workshop-friendly local model path: it wraps a managed
llama.cpp server with Qwen3-0.6B GGUF defaults, bounded generation settings, and
non-thinking prompt controls.

Technical Implementation
------------------------

1. Construct ``DemoLLMClient`` through public APIs so the managed llama.cpp server
   lifecycle remains hidden from workshop scripts.
2. Send one ``LLMRequest`` with a short design-research prompt and bounded output.
3. Print the client description, server snapshot, and normalized response payload
   for repeatable docs and smoke-test output.

.. mermaid::

   1. Construct ``DemoLLMClient`` through public APIs so the managed llama.cpp server
      lifecycle remains hidden from workshop scripts.
   2. Send one ``LLMRequest`` with a short design-research prompt and bounded output.
   3. Print the client description, server snapshot, and normalized response payload
      for repeatable docs and smoke-test output.

.. literalinclude:: ../../../examples/clients/demo_client.py
   :language: python
   :lines: 25-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/demo_client.py

The example prints one JSON payload with client configuration and a single
response. Under deterministic example tests, the model call is monkeypatched.

References
----------

- `Qwen3-0.6B model card <https://huggingface.co/Qwen/Qwen3-0.6B>`_
- `Qwen3-0.6B GGUF model card <https://huggingface.co/Qwen/Qwen3-0.6B-GGUF>`_
- `Qwen llama.cpp local run guide <https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html>`_
