VS Code Hello World
===================

Source: ``examples/agents/vscode_hello_world.py``

Introduction
------------

This example is intentionally self-contained so the VS Code launch configuration can run on a fresh checkout
without a live model server. It still exercises the public ``DirectLLMCall`` API, which makes it useful for
verifying that the virtual environment, debugger, editable install, and ``PYTHONPATH`` wiring are all correct.

Technical Implementation
------------------------

1. Define a tiny local client that implements ``generate(...)`` and returns a deterministic ``LLMResponse``.
2. Construct ``DirectLLMCall`` using only the public top-level package API.
3. Run one prompt through the direct-call path and collect the normalized execution summary.
4. Print JSON output so the VS Code debugger and terminal show one obvious success signal.

.. mermaid::

   flowchart LR
       A["Press F5 in VS Code"] --> B["vscode_hello_world.py"]
       B --> C["_HelloWorldLLMClient.generate(...)"]
       B --> D["DirectLLMCall.run(...)"]
       C --> D
       D --> E["ExecutionResult.summary()"]
       E --> F["JSON output in integrated terminal"]

.. literalinclude:: ../../../examples/agents/vscode_hello_world.py
   :language: python
   :lines: 46-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/agents/vscode_hello_world.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "Hello from the VS Code onboarding example.",
     "terminated_reason": null,
     "error": null
   }

References
----------

- `VS Code Python environments <https://code.visualstudio.com/docs/python/environments>`_
- `Python virtual environments <https://docs.python.org/3/library/venv.html>`_
- `VS Code debugging <https://code.visualstudio.com/docs/debugtest/debugging>`_
