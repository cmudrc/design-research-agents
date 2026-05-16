Quickstart
==========

This example shows the shortest meaningful path through
``design-research-agents``.

Requires Python 3.12+.

.. note::

   If you want a step-by-step editor workflow for creating a virtual
   environment, installing the published package, and running a first script,
   see :doc:`vscode_setup`.

1. Install
----------

Install the published package with a Python 3.12+ interpreter:

.. code-block:: bash

   python3 -m pip install design-research-agents

Windows note:
If ``python`` or ``pip`` resolve to an older interpreter, use
``py -3.12 -m pip install design-research-agents`` and
``py -3.12 -m venv .venv`` for any virtual-environment setup.

Or install from source:

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

2. Minimal Runnable Example
---------------------------

This snippet requires an OpenAI-compatible endpoint (local or remote).

.. code-block:: python

   from design_research_agents import DirectLLMCall, OpenAICompatibleHTTPLLMClient

   with OpenAICompatibleHTTPLLMClient(
       base_url="http://127.0.0.1:8001/v1",
       default_model="qwen2-1.5b-q4",
   ) as llm_client:
       agent = DirectLLMCall(llm_client=llm_client)
       result = agent.run("List three interview themes about onboarding friction.")
       print(result.output)

3. What Happened
----------------

You instantiated a concrete participant (``DirectLLMCall``), executed one run
through a configured backend client, and received structured output that can be
traced and compared in later studies.

4. Where To Go Next
-------------------

- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`examples/index`
- :doc:`api`

Ecosystem Note
--------------

In a typical study, ``design-research-agents`` provides executable
participants, ``design-research-problems`` supplies the task,
``design-research-experiments`` defines the study structure, and
``design-research-analysis`` interprets the resulting records.
