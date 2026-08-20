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

   python -m pip install design-research-agents

Windows note:
If ``python`` or ``pip`` resolve to an older interpreter, use
``py -3.12 -m pip install design-research-agents`` and
``py -3.12 -m venv .venv`` for any virtual-environment setup.

Or install from source:

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .

2. Minimal Runnable Example
---------------------------

This first example is offline and deterministic. It exercises
``DirectLLMCall`` through the minimal runtime methods that this participant
uses, without a model download, API key, or running endpoint. The stub is
intentionally smaller than the complete ``LLMClient`` interface implemented
by the packaged, type-checked backends.

.. code-block:: python

   from design_research_agents import DirectLLMCall, LLMRequest, LLMResponse


   class LocalStubClient:
       def generate(self, request: LLMRequest) -> LLMResponse:
           return LLMResponse(
               text=f"Offline response to: {request.messages[-1].content}",
               model="local-stub",
               provider="local-stub",
           )

       def default_model(self) -> str:
           return "local-stub"

       def close(self) -> None:
           return None


   agent = DirectLLMCall(llm_client=LocalStubClient())
   result = agent.run("List three interview themes about onboarding friction.")
   print(result.final_output)

3. What Happened
----------------

You instantiated a concrete participant (``DirectLLMCall``), executed one run
through its small runtime client boundary, and received structured output that
can be traced and compared in later studies. Replace the stub with a complete
client from :doc:`llm_clients/index` when you are ready to use a real model
backend or type-check against the complete ``LLMClient`` interface.

4. Where To Go Next
-------------------

- :doc:`concepts`
- :doc:`typical_workflow`
- :doc:`examples/index`
- :doc:`api`

Ecosystem Note
--------------

In a typical study, ``design-research-agents`` provides executable
participants, `design-research-problems
<https://cmudrc.github.io/design-research-problems/>`_ supplies the task,
`design-research-experiments
<https://cmudrc.github.io/design-research-experiments/>`_ defines and
orchestrates the study, and `design-research-analysis
<https://cmudrc.github.io/design-research-analysis/>`_ interprets the resulting
records.
