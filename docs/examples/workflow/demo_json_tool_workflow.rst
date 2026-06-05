Demo JSON Tool Workflow
=======================

Source: ``examples/workflow/demo_json_tool_workflow.py``

Introduction
------------

This workshop-sized example uses ``DemoLLMClient`` with a JSON-mode
``MultiStepAgent`` so Qwen3-0.6B can select a real core tool and then finish
with a structured answer.

Technical Implementation
------------------------

1. Create a ``Toolbox`` with core tools and a ``DemoLLMClient`` using Qwen3-0.6B
   GGUF defaults.
2. Configure ``MultiStepAgent(mode="json")`` with ``text.word_count`` as the
   only allowed runtime tool.
3. Run a short prompt that forces one tool call and one final-answer step, then
   print the normalized execution summary.

.. mermaid::

   1. Create a ``Toolbox`` with core tools and a ``DemoLLMClient`` using Qwen3-0.6B
      GGUF defaults.
   2. Configure ``MultiStepAgent(mode="json")`` with ``text.word_count`` as the
      only allowed runtime tool.
   3. Run a short prompt that forces one tool call and one final-answer step, then
      print the normalized execution summary.

.. literalinclude:: ../../../examples/workflow/demo_json_tool_workflow.py
   :language: python
   :lines: 26-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/demo_json_tool_workflow.py

The example prints an ``ExecutionResult.summary()`` payload. Under deterministic
example tests, the model calls are monkeypatched to avoid starting llama.cpp.

References
----------

- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `Toolformer <https://arxiv.org/abs/2302.04761>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
