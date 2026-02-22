Direct Llm Call
===============

Source: ``examples/agents/direct_llm_call.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/direct_llm_call.py

Motivation
----------

Run one traced ``DirectLLMCall`` for an engineering-design prompt.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Prompt"] --> B["Agent run"]
       B --> C["direct llm call output"]
       C --> D["JSON payload and trace"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `agents` use-cases and run `direct_llm_call`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/agents/direct_llm_call.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/agents/direct_llm_call.py
   :language: python
   :linenos:
