Openai Service Client
=====================

Source: ``examples/clients/openai_service_client.py``

Run Command
-----------

.. code-block:: bash

   PYTHONPATH=src python3 examples/clients/openai_service_client.py

Motivation
----------

Run a traced representative ``OpenAIServiceLLMClient`` chat call.

Diagram
-------

.. mermaid::

   flowchart LR
       A["Client config"] --> B["LLMRequest"]
       B --> C["openai service client response"]
       C --> D["Describe and trace metadata"]

Technical Walkthrough
---------------------

1. Configure the runtime surface for `clients` use-cases and run `openai_service_client`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
----------------

- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
----------

Run with `PYTHONPATH=src python3 examples/clients/openai_service_client.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.

Source Code
-----------

.. literalinclude:: ../../../examples/clients/openai_service_client.py
   :language: python
   :linenos:
